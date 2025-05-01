import base64
import io
import time
import random
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import threading
import subprocess
import os
import chardet
import multiprocessing
import numpy as np
import re
import requests
import dash


# ------------------------------------------------------------
# 🌐 Environment & API Key Setup
# ------------------------------------------------------------

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import Counter
from dash import dcc, html, dash_table, Output, Input, State, callback_context
from dash import no_update

os.environ["SERPAPI_KEY"] = "SERPAPI_KEY"

# ------------------------------------------------------------
# 🧠 State Variables & Globals
# ------------------------------------------------------------

uploaded_people = []
start_time = None
last_result_time = None
MAX_RETRIES = 2
retry_attempts = {}
final_table_ready = None
finalized_table_data = None

# ------------------------------------------------------------
# 📋 Logging Setup
# ------------------------------------------------------------

def open_log_file():
    try:
        os.startfile(log_file_path)
    except Exception as e:
        print(f"⚠️ Failed to open log file: {e}")

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "searchlog.txt")
open(log_file_path, "w").close()

if not os.path.exists(log_file_path):
    raise FileNotFoundError(f"Expected log file not found at: {log_file_path}")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    ]
)

# ------------------------------------------------------------
# 🧵 Thread Pool, Executor, and Model Setup
# ------------------------------------------------------------

logger = logging.getLogger("LinkedInScraper")
logger.info("🔁 Logger started — appending to existing searchlog.txt")
sentence_model = SentenceTransformer("all-mpnet-base-v2")
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", grouped_entities=True)
available_threads = os.cpu_count() or multiprocessing.cpu_count()
max_threads = min(8, max(4, int(available_threads * 0.75)))
executor = ThreadPoolExecutor(max_workers=max_threads)
pending_tasks = []
completed_results = []
total_tasks = 0
task_lock = threading.Lock()
scraping_active = False
start_scrape_time = None

# ------------------------------------------------------------
# 🌍 Chrome WebDriver (Selenium) Setup
# ------------------------------------------------------------

DRIVER_PATH = ChromeDriverManager().install()
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0")
    service = Service(DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(15)
    return driver

# ------------------------------------------------------------
# 🔍 Search Query & Matching Logic
# ------------------------------------------------------------

def generate_query(person):
    return f'"{person["First Name"]} {person["Last Name"]}" "{person["University"]}" site:linkedin.com'

def extract_best_title(text):
    if not text:
        return "Not Found"
    for phrase in ["United States", "Professional Profile", "Connections", "LinkedIn"]:
        text = text.replace(phrase, "").strip()
    possible_titles = text.split("|")
    return max(possible_titles, key=lambda t: fuzz.token_set_ratio(t, text), default="Not Found").strip()

# ------------------------------------------------------------
# 🤖 Named Entity Recognition (NER) Helper
# ------------------------------------------------------------

def extract_ner_entities(text):
    entities = ner_pipeline(text)
    locations = [e['word'] for e in entities if e['entity_group'] == 'LOC']
    organizations = [e['word'] for e in entities if e['entity_group'] == 'ORG']
    persons = [e['word'] for e in entities if e['entity_group'] == 'PER']
    return {
        "locations": list(set(locations)),
        "organizations": list(set(organizations)),
        "persons": list(set(persons))
    }

university_map = {
    "KU": "Kean University",
    "RUN": "Rutgers University - Newark",
    "RUNB": "Rutgers University - Newark",
    "WPU": "William Paterson University",
    "FDU": "Fairleigh Dickinson University",
    "MSU": "Montclair State University",
    "NJCU": "New Jersey City University",
    "BC": "Bloomfield College"
}

# ------------------------------------------------------------
# 🛰️ SerpAPI Location & Income Extraction
# ------------------------------------------------------------

def serpapi_extract_location(full_name, university):
    try:
        import requests
        serp_api_key = os.getenv("SERPAPI_KEY")
        if not serp_api_key:
            logger.error("❌ SERPAPI_KEY not set.")
            return "Unknown"

        query = f"{full_name} {university} LinkedIn location"
        params = {
            "q": query,
            "api_key": serp_api_key,
            "engine": "google",
            "num": 3
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        # Mapping short to full names
        city_map = {
            "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas", "ca": "California",
            "co": "Colorado", "ct": "Connecticut", "de": "Delaware", "fl": "Florida", "ga": "Georgia",
            "hi": "Hawaii", "id": "Idaho", "il": "Illinois", "in": "Indiana", "ia": "Iowa",
            "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
            "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi", "mo": "Missouri",
            "mt": "Montana", "ne": "Nebraska", "nv": "Nevada", "nh": "New Hampshire", "nj": "New Jersey",
            "nm": "New Mexico", "ny": "New York", "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio",
            "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
            "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah", "vt": "Vermont",
            "va": "Virginia", "wa": "Washington", "wv": "West Virginia", "wi": "Wisconsin", "wy": "Wyoming"
        }

        city_keywords = list(city_map.values()) + [
            "Chicago", "Boston", "Atlanta", "Seattle", "San Francisco", "Los Angeles", "San Diego",
            "Philadelphia", "Phoenix", "Houston", "Dallas", "Miami", "Las Vegas", "Austin", "Denver",
            "Minneapolis", "Orlando", "San Jose", "Portland", "New Orleans", "Tampa", "Charlotte"
        ]


        for result in data.get("organic_results", []):
            snippet = result.get("snippet", "").lower()
            title = result.get("title", "").lower()

            full_name_lower = full_name.lower()
            university_lower = university.lower()

            # Must contain both full name and university
            if full_name_lower in snippet and university_lower in snippet:
                # Check for city in snippet
                for city in city_keywords:
                    if re.search(rf'\b{re.escape(city.lower())}\b', snippet):
                        return city

                # Check for abbreviation match in snippet
                for short, full in city_map.items():
                    if re.search(rf'\b{re.escape(short)}\b', snippet):
                        return full

            #backup match if at least name is present
            elif full_name_lower in snippet:
                for city in city_keywords:
                    if re.search(rf'\b{re.escape(city.lower())}\b', snippet):
                        return city




    except Exception as e:
        logger.warning(f"⚠️ Failed to extract location with SerpAPI: {e}")
    return "Unknown"

def is_best_match(full_name, title, cosine_threshold=0.4, fuzzy_threshold=0.75):
    if not title or not full_name:
        return False

    full_name_vec = sentence_model.encode(full_name.strip(), convert_to_tensor=True)
    title_vec = sentence_model.encode(title.strip(), convert_to_tensor=True)

    cosine_score = cos_sim(full_name_vec, title_vec).item()
    fuzzy_score = fuzz.token_set_ratio(full_name.strip().lower(), title.strip().lower()) / 100

    logger.info(f"[Similarity] Cosine: {cosine_score:.2f} | Fuzzy: {fuzzy_score:.2f}")
    return cosine_score >= cosine_threshold or fuzzy_score >= fuzzy_threshold

def serpapi_extract_income(title, location="USA"):
    try:
        import requests
        serp_api_key = os.getenv("SERPAPI_KEY")
        if not serp_api_key:
            logger.error("❌ SERPAPI_KEY not set.")
            return "Unknown"

        query = f"{title} average salary in {location}"
        params = {
            "q": query,
            "api_key": serp_api_key,
            "engine": "google",
            "num": 3
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        for result in data.get("organic_results", []):
            snippet = result.get("snippet", "")
            # Look for salary-related keywords near the dollar sign
            context_match = re.search(r"(salary|average|pay|compensation|earnings).{0,40}?\$[0-9,]+", snippet, re.IGNORECASE)
            if context_match:
                dollar_match = re.search(r"\$[0-9,]+", context_match.group())
                if dollar_match:
                    income = dollar_match.group()
                    # Sanity check: if income is greater than $500k, it's probably wrong
                    income_value = int(income.replace("$", "").replace(",", ""))
                    if income_value <= 500000:
                        return income
                    
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract income with SerpAPI: {e}")
    return "Unknown"

def serpapi_search_linkedin_profile(person):

    serp_api_key = os.getenv("SERPAPI_KEY")
    if not serp_api_key:
        logger.error("❌ SERPAPI_KEY not set.")
        return None

    full_name = f"{person['First Name']} {person['Last Name']}"
    university = university_map.get(person["University"], person["University"])
    query = f'"{full_name}" "{university}" site:linkedin.com'

    params = {
        "q": query,
        "api_key": serp_api_key,
        "engine": "google",
        "num": 10
    }

    try:
        logger.info(f"🔍 Querying SerpAPI for: {query}")
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        for result in data.get("organic_results", []):
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "")
            logger.debug(f"🔗 {title} — {url}")

            if is_best_match(full_name, title):
                cosine_score = cos_sim(
                    sentence_model.encode(full_name, convert_to_tensor=True),
                    sentence_model.encode(title, convert_to_tensor=True)
                ).item()
                fuzzy_score = fuzz.token_set_ratio(full_name.lower(), title.lower()) / 100
                best_score = max(cosine_score, fuzzy_score)

                # 🧠 Try NER-based fallback for location extraction
                location = serpapi_extract_location(full_name, university)
                if location == "Unknown":
                    ner_results = extract_ner_entities(snippet)
                    if ner_results["locations"]:
                        location = ner_results["locations"][0]
                    else:
                        location = "USA"


                grad_year = person.get("Graduation Year")
                if not grad_year:
                    grad_year_match = re.search(r"\b(20[0-2][0-9])\b", snippet)
                    grad_year = grad_year_match.group(1) if grad_year_match else "N/A"



                return {
                    "First Name": person["First Name"],
                    "Last Name": person["Last Name"],
                    "University": university,
                    "Graduation Year": grad_year,
                    "LinkedIn Title": extract_best_title(title),
                    "LinkedIn URL": f'<a href="{url}" target="_blank">Open Profile</a>',
                    "Score": f"{int(best_score * 100)}%",
                    "Location (Estimated)": location,
                    "_TitleRaw": title
                }

        logger.warning(f"⚠️ No good result found in SerpAPI for {full_name}")
    except Exception as e:
        logger.error(f"❌ SerpAPI search error: {e}")

    return None

# ------------------------------------------------------------
# 🧪 Main Person Search Logic (SerpAPI + Selenium Fallback)
# ------------------------------------------------------------

def search_person(person, cosine_threshold=0.4, fuzzy_threshold=0.75):
    # Try SerpAPI first
    serp_result = serpapi_search_linkedin_profile(person)
    if serp_result:
        return serp_result  # SerpAPI found a match, return it immediately

    # Fallback to Bing scraping via Selenium if SerpAPI fails
    full_name = f"{person['First Name']} {person['Last Name']}"
    key = f"{full_name} ({person['University']})"
    attempt = retry_attempts.get(key, 0)

    if not person.get("First Name") or not person.get("Last Name") or not person.get("University"):
        logger.warning("⚠️ Skipping entry — missing data.")
        return None

    full_university = university_map.get(person["University"], person["University"])
    query = generate_query({
        "First Name": person["First Name"],
        "Last Name": person["Last Name"],
        "University": full_university
    })

    try:
        driver = create_driver()
        driver.get(f"https://www.bing.com/search?q={query}")
        time.sleep(random.uniform(2, 3))
        elements = driver.find_elements(By.CSS_SELECTOR, "li.b_algo")[:10]

        for elem in elements:
            try:
                title = elem.find_element(By.TAG_NAME, "h2").text.strip()
                url = elem.find_element(By.TAG_NAME, "a").get_attribute("href")
                snippet = elem.find_element(By.CLASS_NAME, "b_caption").text.strip()

                # Try grabbing meta description if available
                meta_description = ""
                try:
                    meta = elem.find_element(By.CLASS_NAME, "b_attribution")
                    meta_description = meta.text.strip()
                except:
                    pass

                location = "Unknown"
                ner_parts = [title, snippet, meta_description, url]
                ner_input = ". ".join([p for p in ner_parts if p])
                logger.warning(f"🌐 Fallback NER input for {full_name}: {ner_input}")
                ner_results = extract_ner_entities(ner_input)

                if ner_results["locations"]:
                    loc_counts = Counter(ner_results["locations"])
                    location = loc_counts.most_common(1)[0][0]
                    logger.info(f"📍 NER location fallback: {location}")

                if ner_results["organizations"]:
                    logger.info(f"🏢 NER detected org: {ner_results['organizations']}")
            except Exception as e:
                logger.warning(f"⚠️ Error extracting element: {e}")
                continue

            if is_best_match(full_name, title, cosine_threshold, fuzzy_threshold):
                cosine_score = cos_sim(
                    sentence_model.encode(full_name, convert_to_tensor=True),
                    sentence_model.encode(title, convert_to_tensor=True)
                ).item()
                fuzzy_score = fuzz.token_set_ratio(full_name.lower(), title.lower()) / 100
                best_score = max(cosine_score, fuzzy_score)
                logger.info(f"✅ Match found: {full_name} → {title}")

                # 🔍 Run NER on the matched title for possible org insights
                ner_results = extract_ner_entities(title)
                if ner_results["organizations"]:
                    logger.info(f"🏢 NER detected possible org: {ner_results['organizations']}")


                location = serpapi_extract_location(full_name, full_university)

                if location == "Unknown":
                    ner_results = extract_ner_entities(snippet)
                    if ner_results["locations"]:
                        location = ner_results["locations"][0]
                    else:
                        location = "USA"

                return {
                    "First Name": person["First Name"],
                    "Last Name": person["Last Name"],
                    "University": full_university,
                    "Graduation Year": person.get("Graduation Year", "N/A"),
                    "LinkedIn Title": extract_best_title(title),
                    "LinkedIn URL": f'<a href="{url}" target="_blank">Open Profile</a>',
                    "Status": "✅ Found",
                    "Score": f"{int(best_score * 100)}%",
                    "Location (Estimated)": location,
                    "_TitleRaw": title
                }


    except Exception as e:
        logger.error(f"❌ Search error for {full_name}: {e}")
        if attempt < MAX_RETRIES:
            retry_attempts[key] = attempt + 1
            logger.warning(f"🔁 Retrying {key} (Attempt {attempt + 1}/{MAX_RETRIES})")
            return search_person(person, cosine_threshold, fuzzy_threshold)
        else:
            logger.error(f"❌ Failed after {MAX_RETRIES} attempts: {key}")
    finally:
        try:
            driver.quit()
        except:
            pass

    return None

# ------------------------------------------------------------
# 💰 Income Finalization Logic
# ------------------------------------------------------------

def finalize_income_estimates(results):
    finalized = []
    for r in results:
        title = r.get("_TitleRaw", r.get("LinkedIn Title", "")).strip()
        location = r.get("Location (Estimated)", "")

        if not title or title.lower() in ["not found", "error", ""]:
            r["Income (Estimated)"] = "Unknown"
        else:
            try:
                income_est = serpapi_extract_income(title, location or "USA")
                r["Income (Estimated)"] = income_est
                logger.info(f"💰 {r['First Name']} {r['Last Name']} income: {income_est}")
            except Exception as e:
                logger.warning(f"⚠️ Income lookup failed for: {title} | {e}")
                r["Income (Estimated)"] = "Unknown"

        if "_TitleRaw" in r:
            del r["_TitleRaw"]
        finalized.append(r)
    return finalized

def complete_results_flow():
    global completed_results
    logger.info("⚙️ Running income estimation finalization...")
    completed_results = finalize_income_estimates(completed_results)
    logger.info("✅ Income queries added to all results.")

# ------------------------------------------------------------
# 📦 Task Queue Management
# ------------------------------------------------------------

def enqueue_tasks(people_list, cosine_val=0.4, fuzzy_val=0.75):
    global pending_tasks, completed_results, total_tasks, scraping_active, start_scrape_time
    pending_tasks.clear()
    completed_results.clear()
    total_tasks = len(people_list)
    scraping_active = True
    start_scrape_time = time.time()
    logger.info(f"🚀 Starting search with {total_tasks} person(s)...")
    for person in people_list:
        future = executor.submit(search_person, person)
        future.add_done_callback(task_done)
        pending_tasks.append(future)

def task_done(future):
    global scraping_active, last_result_time
    with task_lock:
        try:
            result = future.result(timeout=30)
            if result:
                result["Status"] = "✅ Found"
                completed_results.append(result)
                last_result_time = time.time()
                logger.info(f"🔎 Completed: {result['First Name']} {result['Last Name']}")
            else:
                completed_results.append({
                    "First Name": "N/A",
                    "Last Name": "N/A",
                    "University": "N/A",
                    "Graduation Year": "N/A",
                    "LinkedIn Title": "Not Found",
                    "LinkedIn URL": "",
                    "Score": "N/A",
                    "Location (Estimated)": "N/A",
                    "Income (Estimated)": "N/A",
                    "Status": "❌ No Match"
                })
        except Exception as e:
            logger.error(f"❌ Task failed: {e}")
            completed_results.append({
                "First Name": "N/A",
                "Last Name": "N/A",
                "University": "N/A",
                "Graduation Year": "N/A",
                "LinkedIn Title": "Error",
                "LinkedIn URL": "",
                "Score": "N/A",
                "Location (Estimated)": "N/A",
                "Income (Estimated)": "N/A",
                "Status": "❌ Error"
            })

        if len(completed_results) >= total_tasks:
            scraping_active = False
            logger.info("✅ All searches complete.")
            complete_results_flow()

# ------------------------------------------------------------
# 🖼️ Dash App & Layout
# ------------------------------------------------------------

external_stylesheets = [
    {
        'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap',
        'rel': 'stylesheet'
    },
    {
        'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
        'rel': 'stylesheet'
    }
]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# Track retry attempts for each person
retry_attempts = {}
MAX_RETRIES = 2

# ------------------------------------------------------------
# 🧠 Dash Callbacks — Upload, Search, Stop, Restart, Download
# ------------------------------------------------------------

app.layout = html.Div([
    
    html.Div([
        html.H1("LinkedIn Profile Finder", style={
            'color': '#0a66c2',
            'fontWeight': '700',
            'fontSize': '40px',
            'marginBottom': '10px'
        }),
        html.P("Search and match LinkedIn profiles based on university and name.", style={
            'fontSize': '18px',
            'color': '#666'
        }),
        html.A(html.Span([
            html.I(className='fas fa-info-circle', style={'marginRight': '8px'}),
            "How this program works"
        ]),
            href="/assets/42f021a5-7b2b-40eb-99d8-42e07bc11f8d.pdf", # Link to the PDF
            target="_blank", # Open in a new tab
            className='floating-btn info-btn', # Keep existing class for styling
            style={ # Keep existing style
                'fontSize': '16px',
                'padding': '10px 20px',
                'borderRadius': '12px',
                'backgroundColor': '#17a2b8', # Info color
                'color': '#ffffff',
                'border': 'none',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
                'cursor': 'pointer',
                'transition': 'transform 0.3s ease',
                'fontWeight': '500',
                'textDecoration': 'none' # Remove underline from link
            }
        )
    ], style={
        'textAlign': 'center',
        'marginBottom': '40px'
    }),
    
    

    html.Div([
        html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Button(html.Span([
                        html.I(className='fas fa-upload', style={'marginRight': '10px'}),
                        "Upload CSV"
                    ]), className='floating-btn upload-btn', style={
                        'fontSize': '18px',
                        'padding': '16px 36px',
                        'borderRadius': '16px',
                        'backgroundColor': '#ffffff',
                        'color': '#0a66c2',
                        'border': '2px solid #0a66c2',
                        'boxShadow': '0 6px 16px rgba(0,0,0,0.1)',
                        'cursor': 'pointer',
                        'transition': 'transform 0.3s ease',
                        'fontWeight': '600'
                    }),
                    multiple=False
                ),
                html.Button(html.Span([
                    html.I(className='fas fa-sync-alt', style={'marginRight': '10px'}),
                    "Restart"
                ]), id="restart-button", className='floating-btn restart-btn', style={
                    'fontSize': '18px',
                    'padding': '16px 36px',
                    'borderRadius': '16px',
                    'backgroundColor': '#6c757d',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 6px 16px rgba(0,0,0,0.15)',
                    'cursor': 'pointer',
                    'transition': 'transform 0.3s ease',
                    'fontWeight': '600'
                }),
                html.Button(html.Span([
                    html.I(className='fas fa-download', style={'marginRight': '10px'}),
                    "Download Results"
                ]), id="download-button", className='floating-btn download-btn', style={
                    'fontSize': '18px',
                    'padding': '16px 36px',
                    'borderRadius': '16px',
                    'backgroundColor': '#28a745',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 6px 16px rgba(0,0,0,0.15)',
                    'cursor': 'pointer',
                    'transition': 'transform 0.3s ease',
                    'fontWeight': '600'
                }),
            ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'gap': '30px',
                'marginBottom': '30px',
                'flexWrap': 'wrap'
            }),
            html.Div([
                html.Div([
                    html.Label("Manual Entry (Optional):", style={'fontWeight': '500'}),
                    dcc.Input(id='first-name', placeholder='First Name', type='text', style={'marginRight': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='last-name', placeholder='Last Name', type='text', style={'marginRight': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='university', placeholder='University', type='text', style={'marginRight': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='grad-year', placeholder='Graduation Year (Optional)', type='text', style={'marginTop': '10px', 'fontSize': '16px'}),
                ], style={'marginBottom': '20px'}),

                html.Label("Cosine Confidence Threshold (0.30 - 0.90):", style={
                    'fontWeight': '500', 'marginTop': '20px', 'fontSize': '16px'
                }),
                dcc.Slider(
                    id='cosine-threshold',
                    min=0.3, max=0.9, step=0.05, value=0.4,
                    marks={i: f"{i:.2f}" for i in [0.3, 0.4, 0.5, 0.6, 0.75, 0.9]},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False
                ),
                html.Div([
                    html.Label("Limit number of people to search (optional):", style={
                        'fontWeight': '500',
                        'fontSize': '16px',
                        'marginBottom': '8px'
                    }),
                    dcc.Input(
                        id='name-limit',
                        type='number',
                        min=1,
                        step=1,
                        placeholder='Leave blank for all',
                        style={
                            'width': '100%',
                            'maxWidth': '300px',
                            'textAlign': 'center',
                            'marginBottom': '20px',
                            'padding': '10px',
                            'border': '1px solid #ccc',
                            'borderRadius': '6px'
                        }
                    )
                ], style={'marginTop': '30px', 'textAlign': 'center'}),
                html.Label("Fuzzy Matching Threshold (0.60 - 0.95):", style={
                    'fontWeight': '500', 'marginTop': '20px', 'fontSize': '16px'
                }),
                dcc.Slider(
                    id='fuzzy-threshold',
                    min=0.6, max=0.95, step=0.05, value=0.75,
                    marks={i: f"{i:.2f}" for i in [0.6, 0.7, 0.75, 0.85, 0.95]},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False
                ),
                html.Button(html.Span([
                    html.I(className='fas fa-play', style={'marginRight': '10px'}),
                    "Start Search"
                ]), id="search-button", className='floating-btn start-btn', style={
                    'fontSize': '18px',
                    'padding': '16px 36px',
                    'borderRadius': '16px',
                    'backgroundColor': '#0a66c2',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 6px 16px rgba(0,0,0,0.15)',
                    'cursor': 'pointer',
                    'transition': 'transform 0.3s ease',
                    'fontWeight': '600',
                    'marginTop': '30px'
                }),
                
            ], style={
                'marginBottom': '20px',
                'textAlign': 'center',
                'padding': '0 40px'
            }),
            html.Div(id='upload-status', style={'marginBottom': '10px', 'textAlign': 'center', 'fontSize': '16px'}),
            html.Div(id="search-status", style={"marginTop": "10px", "fontWeight": "500", 'textAlign': 'center', 'fontSize': '16px'}),
            html.Div(id="eta-stats", style={'marginTop': '10px', 'textAlign': 'center', 'color': '#888'}),
            dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=False),
        ], style={
            'width': '100%',
            'maxWidth': '800px',
            'margin': '0 auto',
            'textAlign': 'center',
            'padding': '30px',
            'backgroundColor': '#ffffff',
            'borderRadius': '20px',
            'boxShadow': '0 12px 28px rgba(0, 0, 0, 0.1)'
        }),

        html.Div(id="progress-container", children=[
            html.Div("Progress:", style={
                'fontSize': '16px',
                'marginTop': '30px',
                'marginBottom': '10px',
                'textAlign': 'center'
            }),
            html.Div(style={
                'width': '100%',
                'backgroundColor': '#e0e0e0',
                'borderRadius': '10px',
                'maxWidth': '500px',
                'margin': '0 auto',
                'overflow': 'hidden'
            }, children=[
                html.Div(id="progress-bar", style={
                    "height": "30px",
                    "width": "0%",
                    "backgroundColor": "#0a66c2",
                    "color": "white",
                    "textAlign": "center",
                    "lineHeight": "30px",
                    'borderRadius': '10px',
                    'transition': 'width 0.5s ease-in-out',
                    'fontWeight': '600'
                })
            ]),
            html.Div([
                html.Button(html.Span([
                    html.I(className='fas fa-stop', style={'marginRight': '10px'}),
                    "Stop"
                ]), id="stop-button", className='floating-btn stop-btn', style={
                    'fontSize': '18px',
                    'padding': '16px 36px',
                    'borderRadius': '16px',
                    'backgroundColor': '#dc3545', # Red color for stop
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 6px 16px rgba(0,0,0,0.15)',
                    'cursor': 'pointer',
                    'transition': 'transform 0.3s ease',
                    'fontWeight': '600',
                    'marginTop': '20px' # Add margin for spacing
                })
            ], style={'textAlign': 'center'}) # Center the button
        ]),
        html.Div("🔎 Manual Mode", id="manual-mode-label", style={
            'textAlign': 'center',
            'color': '#0a66c2',
            'fontWeight': 'bold',
            'marginTop': '10px',
            'display': 'none'
}),

    ]),
    html.Div(id="results-anchor"),


    html.Div([
        dash_table.DataTable(
            id='results-table',
            columns=[
                {"name": "No.", "id": "No."},
                {"name": "First Name", "id": "First Name"},
                {"name": "Last Name", "id": "Last Name"},
                {"name": "University", "id": "University"},
                {"name": "Graduation Year", "id": "Graduation Year"},
                {"name": "LinkedIn Title", "id": "LinkedIn Title"},
                {"name": "LinkedIn URL", "id": "LinkedIn URL", "presentation": "markdown"},
                {"name": "Status", "id": "Status"},
                {"name": "Confidence Score", "id": "Score"},
                {"name": "Location (Estimated)", "id": "Location (Estimated)"},
                {"name": "Income (Estimated)", "id": "Income (Estimated)"},
            ],

            data=[],
            page_action='native',
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto', 'marginTop': '40px'},
            style_header={
                'backgroundColor': '#0a66c2',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontFamily': 'Roboto, sans-serif',
                'fontSize': '16px'
            },
            markdown_options={"html": True}
        ),
        dcc.Download(id="download-dataframe-csv")
    ], style={
        'marginTop': '40px',
        'padding': '30px',
        'backgroundColor': '#ffffff',
        'borderRadius': '20px',
        'boxShadow': '0 12px 28px rgba(0, 0, 0, 0.1)',
        'maxWidth': '1200px',
        'marginLeft': 'auto',
        'marginRight': 'auto'
    }),
    html.Footer(
        html.A(
            html.Span([
                html.I(className='fas fa-info-circle', style={'marginRight': '8px'}),
                "Ethical Use of Web Scraping Technologies"
            ]),
            href="/assets/Ethical Use of Web Scraping Technologies.pdf",
            target="_blank",
            className='floating-btn info-btn',
            style={
                'fontSize': '16px',
                'padding': '10px 20px',
                'borderRadius': '12px',
                'backgroundColor': '#28a745',
                'color': '#ffffff',
                'border': 'none',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
                'cursor': 'pointer',
                'transition': 'transform 0.3s ease',
                'fontWeight': '500',
                'textDecoration': 'none'
            }
        ),
        style={
            'textAlign': 'center',
            'padding': '20px 0',
            'backgroundColor': '#f3f2ef',
            'marginTop': '40px'
        }
    )

    
], style={
    'padding': '50px 20px',
    'fontFamily': 'Roboto, sans-serif',
    'backgroundColor': '#f3f2ef'
})

app.clientside_callback(
    """
    function(n_clicks) {
        const anchor = document.getElementById('results-anchor');
        if (anchor) {
            anchor.scrollIntoView({ behavior: 'smooth' });
        }
        return '';
    }
    """,
    Output("results-anchor", "children"),
    Input("search-button", "n_clicks"),
    prevent_initial_call=True
)

app.index_string = app.index_string.replace('</head>', '''<style>
.floating-btn:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
}
</style>
</head>''')






# Custom index_string with updated CSS for a modern and responsive design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>LinkedIn Search Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            /* Base Styles */
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 0;
                color: #333;
            }
            .app-container {
                max-width: 1500px;
                margin: auto;
                padding: 20px;
                position: relative;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .app-title {
                color: #0072b1;
                font-size: 40px;
                font-weight: 700;
                margin: 0;
            }
            /* Card Styles */
            .card {
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .section-title {
                font-size: 30px;
                font-weight: 500;
                color: #333;
                margin-bottom: 15px;
            }
            /* Input & Upload Styles */
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #0072b1;
                background-color: #f0f8ff;
            }
            .upload-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
            }
            .upload-icon {
                font-size: 24px;
                color: #0072b1;
            }
            .upload-text {
                color: #6c757d;
            }
            .status-message {
                margin-top: 8px;
                font-size: 18px;
                font-weight: 500;
                min-height: 20px;
            }
            .input-container {
                display: grid;
                gap: 10px;
                margin-bottom: 15px;
            }
            .input-group {
                display: flex;
                flex-direction: column;
            }
            .input-group label {
                margin-bottom: 6px;
                font-size: 20px;
                font-weight: 500;
            }
            .input-field {
                padding: 14px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            .input-field:focus {
                border-color: #0072b1;
                outline: none;
            }
            .action-container {
                margin-top: 15px;
            }
            /* Button Styles */
            .btn {
                padding: 12px 16px;
                border: none;
                border-radius: 4px;
                font-size: 18px;
                font-weight: 500;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .btn-primary {
                background-color: #0072b1;
                color: #fff;
            }
            .btn-primary:hover {
                background-color: #005b8e;
            }
            .btn-secondary {
                background-color: #e2e2e2;
                color: #333;
            }
            .btn-secondary:hover {
                background-color: #e2e2e2;
            }
            .btn-download {
                background-color: #28a745;
                color: white;
            }
            .btn-download:hover {
                background-color: #218838;
            }
            .download-container {
                margin-top: 15px;
                display: flex;
                justify-content: flex-end;
            }
            /* Layout Styles */
            .flex-container {
                display: flex;
                gap: 20px;
            }
            .left-panel {
                flex-shrink: 0;
            }
            .right-panel {
                flex-grow: 1;
            }
            /* Responsive Styles */
            @media screen and (max-width: 768px) {
                .flex-container {
                    flex-direction: column;
                }
                .left-panel, .right-panel {
                    width: 100%;
                    padding: 10px;
                }
            }
            .thread-info {
                position: absolute;
                bottom: 10px;
                right: 20px;
                color: #0072b1;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ------------------------------------------------------------
# 🔄 Dash Callback Context
# ------------------------------------------------------------

ctx = dash.callback_context


@app.callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def parse_upload(contents, filename):
    global uploaded_people
    if not contents or not filename:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    encoding = chardet.detect(decoded)['encoding']
    try:
        df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
        uploaded_people = df.to_dict(orient='records')
        logger.info(f"📥 Uploaded {filename} with {len(uploaded_people)} rows.")
        return f"✅ Uploaded {filename} with {len(uploaded_people)} row(s)."
    except Exception as e:
        err = f"❌ Upload failed: {e}"
        logger.error(err)
        return err

@app.callback(
    Output("results-table", "data", allow_duplicate=True),
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-bar", "children", allow_duplicate=True),
    Output("search-status", "children", allow_duplicate=True),
    Output("eta-stats", "children", allow_duplicate=True),
    Output("interval", "disabled", allow_duplicate=True),
    Output("manual-mode-label", "style", allow_duplicate=True),
    Input("search-button", "n_clicks"),
    Input("interval", "n_intervals"),
    State("first-name", "value"),
    State("last-name", "value"),
    State("university", "value"),
    State("grad-year", "value"),
    State("cosine-threshold", "value"),
    State("fuzzy-threshold", "value"),
    State("name-limit", "value"),
    prevent_initial_call=True
)
def update_table(search_clicks, interval, fname, lname, university, grad_year, cosine_val, fuzzy_val, name_limit):
    global scraping_active, completed_results, total_tasks, start_scrape_time, final_table_ready, finalized_table_data

    ctx = dash.callback_context
    triggered = ctx.triggered_id

    if triggered == "search-button":
        final_table_ready = False
        finalized_table_data = []

        manual_mode_style = {"display": "none"}

        if fname and lname and university:
            person = {
                "First Name": fname,
                "Last Name": lname,
                "University": university
            }
            if grad_year:
                person["Graduation Year"] = grad_year
            people = [person]
            manual_mode_style = {"display": "block"}  # 👈 Enable the label if manually entered

        elif uploaded_people:
            people = uploaded_people
        else:
            return no_update, no_update, no_update, "⚠️ No data provided.", no_update, True, {"display": "none"}

        if name_limit and isinstance(name_limit, int) and name_limit > 0:
            people = people[:name_limit]

        start_scrape_time = time.time()
        enqueue_tasks(people, cosine_val, fuzzy_val)
        return [], {"width": "0%"}, "0%", "🔎 Searching...", "ETA calculating...", False, manual_mode_style

    elif triggered == "interval" and scraping_active:
        percent = int((len(completed_results) / total_tasks) * 100)
        elapsed = time.time() - start_scrape_time
        speed = len(completed_results) / elapsed if elapsed > 0 else 0
        remaining = total_tasks - len(completed_results)
        eta = remaining / speed if speed > 0 else 0

        results = completed_results.copy()
        for i, r in enumerate(results):
            r["No."] = i + 1
            r.setdefault("Income (Estimated)", "Unknown")
            r.setdefault("Graduation Year", "N/A")
            r.setdefault("Location (Estimated)", "Unknown")
            r.setdefault("LinkedIn URL", "")
            r.setdefault("Score", "N/A")

        if last_result_time and time.time() - last_result_time > 30:
            logger.warning("⏳ No progress for 30+ seconds — recommend restarting.")
            return (
                results,
                {"width": f"{percent}%", "height": "30px", "backgroundColor": "#ffc107",
                 "color": "#000", "textAlign": "center", "lineHeight": "30px",
                 'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'},
                f"{percent}%",
                "⚠️ No progress detected. Please consider pressing Restart.",
                f"⏱ ETA: stalled | Speed: {speed:.2f}/sec",
                False,
                {"display": "none"}
            )

        return (
            results,
            {"width": f"{percent}%", "height": "30px", "backgroundColor": "#0a66c2",
             "color": "white", "textAlign": "center", "lineHeight": "30px",
             'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'},
            f"{percent}%",
            "⏳ Running...",
            f"⏱ ETA: {int(eta)}s | Speed: {speed:.2f}/sec",
            False,
            {"display": "none"}
        )

    elif not scraping_active and completed_results:

        if not final_table_ready:
            logger.info("📊 Finalizing income data before displaying table...")
            finalized_table_data = finalize_income_estimates(completed_results.copy())
            for i, r in enumerate(finalized_table_data):
                r["No."] = i + 1
                r.setdefault("Location (Estimated)", "Unknown")
                r.setdefault("LinkedIn URL", "")
                r.setdefault("Score", "N/A")
                r.setdefault("Status", "❌ No Match")
            final_table_ready = True

        return finalized_table_data, {
            "width": "100%", "height": "30px", "backgroundColor": "#0a66c2",
            "color": "white", "textAlign": "center", "lineHeight": "30px",
            'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'
        }, "100%", "✅ Search complete.", "Done.", True, {"display": "none"}

    return no_update, no_update, no_update, no_update, no_update, no_update, {"display": "none"}



@app.callback(
    Output("results-table", "data", allow_duplicate=True),
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-bar", "children", allow_duplicate=True),
    Output("search-status", "children", allow_duplicate=True),
    Output("interval", "disabled", allow_duplicate=True),
    Input("stop-button", "n_clicks"),
    Input("restart-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_stop_restart(stop_clicks, restart_clicks):
    global scraping_active, completed_results, total_tasks, pending_tasks

    triggered_id = ctx.triggered_id

    if triggered_id == "stop-button":
        scraping_active = False
        logger.info("🛑 Search manually stopped.")

        # ❌ Cancel any queued (not yet running) tasks
        for task in pending_tasks:
            if not task.done():
                task.cancel()

        return no_update, no_update, no_update, "🛑 Search stopped.", True

    elif triggered_id == "restart-button":
        completed_results.clear()
        total_tasks = 0
        pending_tasks.clear()
        logger.info("🔁 Search restarted.")

        return [], {"width": "0%"}, "0%", "🔁 Ready for new search.", True

    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n):
    if not completed_results:
        return no_update
    df = pd.DataFrame(completed_results)
    return dcc.send_data_frame(df.to_csv, "linkedin_results.csv", index=False)

if __name__ == '__main__':
    def open_browser():
        chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        url = "http://127.0.0.1:8050/"
        try:
            subprocess.Popen([chrome_path, "--new-window", "--app=" + url])
        except Exception:
            webbrowser.open_new(url)

    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
