import argparse
import logging
import random
import time
from pathlib import Path
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import json 
import datetime 
import os 
import httpx 
import math  
import re 
from email_client import send_email_alert # Import pour les alertes

# Imports ReportLab (consolidés)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader 

# Imports d'Export/Web Scraping (inutilisés en mode JSON local, mais gardés)
import gspread
from google.oauth2.service_account import Credentials
import requests
from bs4 import BeautifulSoup

# Importation des constantes de configuration
try:
    from config import (
        OUTPUT_DIR, 
        LOG_FILE, 
        DEFAULT_LISTINGS, 
        GOOGLE_SHEET_NAME, 
        CREDENTIALS_FILE,
        SEUIL_PRIX_HAUT, 
        NOTE_MIN_RECOMMANDATION, 
        FACTEUR_OCCUPATION_FAIBLE
    )
except ImportError:
    print("FATAL ERROR: Le fichier config.py est introuvable. Veuillez le créer et le remplir.")
    exit(1)

# --- CONFIGURATION GLOBALE ET CONSTANTES ---
DB_NAME = "airbnb_history.db"
EVENT_PREDICTION_DAYS = 60

# Équipements "Actionnables"
SOFT_KEYWORDS = ['wifi', 'climatisation', 'clim', 'parking', 'nespresso', 'fibre']
# Équipements "Structurels"
HARD_KEYWORDS = ['piscine', 'balcon', 'terrasse', 'vue mer', 'plage'] 

# Base de données des Points d'Intérêt (POI) pour Cannes (Statique)
CANNES_POIS = {
    "Palais des Festivals": {"lat": 43.5513, "lon": 7.0174, "keyword": "Palais", "alias": ["congrès", "festival"]},
    "Gare de Cannes": {"lat": 43.5537, "lon": 7.0199, "keyword": "Gare", "alias": ["train"]},
    "Plage de la Croisette": {"lat": 43.5495, "lon": 7.0279, "keyword": "Plage", "alias": ["croisette", "mer"]},
    "Le Suquet (Vieille Ville)": {"lat": 43.5505, "lon": 7.0108, "keyword": "Suquet", "alias": ["vieille ville", "port"]}
}
# ----------------------------------------------------


# --- Fonctions de Logging ---
Path(OUTPUT_DIR).mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Fonctions Utilitaires (SQLite, Maths) ---

def setup_database():
    """Crée les tables 'analysis_runs' (historique) et 'clients' (profils) avec GPS et SEO."""
    conn = sqlite3.connect(DB_NAME)
    try:
        # 💡 CORRECTION : Ajout de client_id à l'historique
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER, 
                city TEXT NOT NULL,
                num_listings INTEGER NOT NULL,
                date_run TEXT NOT NULL,
                avg_price REAL,
                avg_revenue_monthly REAL,
                avg_profit_net REAL,
                FOREIGN KEY (client_id) REFERENCES clients(client_id)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name TEXT NOT NULL,
                target_property_name TEXT,
                city TEXT NOT NULL,
                target_price REAL,
                target_size REAL,
                target_rating REAL,
                target_occupancy REAL,
                monthly_fixed_costs REAL,
                platform_fee_pct REAL DEFAULT 0.15,
                cleaning_fee_pct REAL DEFAULT 0.10,
                current_title TEXT, 
                current_description TEXT,
                client_email TEXT,
                target_latitude REAL, 
                target_longitude REAL,
                cleaning_fee_per_stay REAL DEFAULT 50.0,
                avg_stay_duration INTEGER DEFAULT 4,
                taxe_de_sejour_par_nuit REAL DEFAULT 1.50,
                tva_services_pct REAL DEFAULT 0.20,
                amenities_list TEXT
            );
        """)
        conn.commit()
        logger.info("Base de données et tables (analysis_runs, clients) vérifiées.")
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la base de données : {e}")
    finally:
        conn.close()

def save_to_sqlite(df, client_id, city):
    """Sauvegarde le résumé de l'analyse (concurrents) dans SQLite."""
    if df.empty: return

    try:
        conn = sqlite3.connect(DB_NAME)
        
        competitors_df = df[df['is_target'] == False]
        if competitors_df.empty:
            avg_price, avg_revenue_monthly, avg_profit_net = 0, 0, 0
        else:
            avg_price = competitors_df['price_per_night'].mean()
            avg_revenue_monthly = competitors_df['revenue_monthly'].mean()
            avg_profit_net = competitors_df['profit_net_monthly'].mean()

        summary_data = {
            'client_id': client_id, # 💡 CORRECTION : Ajout du client_id
            'city': city,
            'num_listings': len(df),
            'date_run': time.strftime("%Y-%m-%d %H:%M:%S"),
            'avg_price': avg_price,
            'avg_revenue_monthly': avg_revenue_monthly,
            'avg_profit_net': avg_profit_net
        }
        
        conn.execute("""
            INSERT INTO analysis_runs (client_id, city, num_listings, date_run, avg_price, avg_revenue_monthly, avg_profit_net)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_data['client_id'], summary_data['city'], summary_data['num_listings'], summary_data['date_run'],
            summary_data['avg_price'], summary_data['avg_revenue_monthly'], summary_data['avg_profit_net']
        ))
        conn.commit()
        logger.info(f"Résumé de l'analyse (concurrents) enregistré dans {DB_NAME} pour client {client_id}.")
        print(f"💾 Résumé de l'analyse enregistré dans {DB_NAME}.")

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement SQLite : {e}", exc_info=True)
        print("❌ Échec de l'enregistrement dans la base de données.")
    finally:
        conn.close()

def save_to_gsheet(df, client_name):
    """Exporte les données analysées vers Google Sheets, dans un onglet dédié au client."""
    if df.empty: return

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        
        spreadsheet = client.open(GOOGLE_SHEET_NAME)
        
        try:
            worksheet = spreadsheet.worksheet(client_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=client_name, rows="100", cols="20")
            
        df_to_write = df.fillna(0)
        
        worksheet.clear()
        worksheet.append_row(list(df_to_write.columns))
        worksheet.append_rows(df_to_write.values.tolist())

        logger.info(f"Données mises à jour dans l'onglet '{client_name}' de la Google Sheet !")
        print(f"📊 Données mises à jour dans l'onglet '{client_name}' !")

    except Exception as e:
        logger.error(f"❌ Impossible d'exporter vers Google Sheets. Erreur: {e}", exc_info=True)
        print("❌ Impossible de se connecter/écrire à Google Sheets. Vérifiez 'credentials.json'.")


# ----------------------------------------------------
# --- Fonctions de Calculs et Analyse ---
# ----------------------------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points GPS (en km) en utilisant la formule de Haversine.
    """
    R = 6371 # Rayon de la Terre en kilomètres
    
    if None in [lat1, lon1, lat2, lon2] or 0 in [lat1, lon1, lat2, lon2]:
        return float('inf') # Retourne "infini" si les données GPS sont manquantes
        
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def calculer_revenus(df):
    """Calcule les revenus estimés (Brut)."""
    if df.empty: return df
    df = df.assign(
        revenue_daily=df["price_per_night"] * df["occupancy"],
        revenue_monthly=lambda x: x["revenue_daily"] * 30,
        revenue_annual=lambda x: x["revenue_monthly"] * 12
    )
    return df

def calculer_profit_net(df, params: dict):
    """
    Calcule le profit net mensuel estimé en utilisant les coûts personnalisés 
    (rotation, taxes, etc.).
    """
    if df.empty: return df
        
    FRAIS_PLATEFORME_PCT = params.get('platform_fee_pct', 0.15)
    TVA_SERVICES_PCT = params.get('tva_services_pct', 0.20) 
    CHARGES_MENSUELLES_FIXES = params.get('monthly_fixed_costs', 300.0)
    CLEANING_FEE_PER_STAY = params.get('cleaning_fee_per_stay', 50.0)
    AVG_STAY_DURATION_NIGHTS = params.get('avg_stay_duration', 4)
    TAXE_DE_SEJOUR_PAR_NUIT = params.get('taxe_de_sejour_par_nuit', 1.50)

    df['nuits_louees_par_mois'] = df['occupancy'] * 30
    
    if AVG_STAY_DURATION_NIGHTS > 0:
        df['estim_sejours_par_mois'] = df['nuits_louees_par_mois'] / AVG_STAY_DURATION_NIGHTS
    else:
        df['estim_sejours_par_mois'] = 0
        
    df['cost_platform_monthly'] = df['revenue_monthly'] * FRAIS_PLATEFORME_PCT
    df['cost_cleaning_total'] = df['estim_sejours_par_mois'] * CLEANING_FEE_PER_STAY
    df['cost_taxes_total'] = df['nuits_louees_par_mois'] * TAXE_DE_SEJOUR_PAR_NUIT
    
    df['profit_net_monthly'] = (
        df['revenue_monthly'] - 
        df['cost_platform_monthly'] - 
        df['cost_cleaning_total'] -   
        df['cost_taxes_total'] -      
        CHARGES_MENSUELLES_FIXES
    )
    
    df['profit_net_annual'] = df['profit_net_monthly'] * 12
    
    return df

def analyser_concurrentiel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'écart par rapport à la moyenne du marché CIBLE (segmenté par taille ET distance).
    """
    if df.shape[0] < 1: return df
    
    if True not in df['is_target'].values:
        logger.warning("Aucune propriété cible ('is_target' == True) trouvée pour l'analyse concurrentielle.")
        return df
        
    target_index = df[df['is_target'] == True].index[0]
    target_property = df.loc[target_index]
    
    target_lat = target_property.get('latitude') 
    target_lon = target_property.get('longitude')
    target_size = target_property.get('size', 50) 
    
    size_tolerance = 15
    distance_limit_km = 2.0 
    
    min_size = target_size - size_tolerance
    max_size = target_size + size_tolerance

    competitors_market = df[df['is_target'] == False].copy()
    
    if competitors_market.empty:
        logger.warning("Aucun concurrent trouvé (DataFrame vide).")
        df.loc[target_index, 'prix_moyen_concurrents'] = target_property['price_per_night']
        df.loc[target_index, 'ecart_prix_vs_moyenne'] = 0
        df.loc[target_index, 'ecart_profit_net_vs_moyenne'] = 0
        return df

    # Filtrage par Taille
    segmented_market = competitors_market[
        (competitors_market['size'] >= min_size) & 
        (competitors_market['size'] <= max_size)
    ]
    
    # Filtrage par GPS
    if target_lat is not None and target_lon is not None:
        if 'latitude' in segmented_market.columns and 'longitude' in segmented_market.columns:
            segmented_market['distance_km'] = segmented_market.apply(
                lambda row: calculate_distance(target_lat, target_lon, row.get('latitude'), row.get('longitude')), 
                axis=1
            )
            segmented_market = segmented_market[
                (segmented_market['distance_km'] <= distance_limit_km)
            ]
            logger.info(f"Segmentation GPS/Taille : {len(segmented_market)} concurrents trouvés dans un rayon de {distance_limit_km}km.")
        else:
            logger.warning("Colonnes GPS concurrentes manquantes. Filtration uniquement par taille.")
    else:
        logger.warning("Coordonnées client manquantes. Filtration uniquement par taille.")
        
    
    if segmented_market.empty:
        logger.warning(f"Aucun concurrent comparable trouvé (Taille/GPS). Utilisation du marché global.")
        segmented_market = competitors_market 

    if segmented_market.empty:
        market_avg_price = 0
        market_avg_profit = 0
    else:
        market_avg_price = segmented_market['price_per_night'].mean()
        market_avg_profit = segmented_market['profit_net_monthly'].mean()

    df.loc[target_index, 'prix_moyen_concurrents'] = market_avg_price
    df.loc[target_index, 'ecart_prix_vs_moyenne'] = df.loc[target_index, 'price_per_night'] - market_avg_price
    df.loc[target_index, 'ecart_profit_net_vs_moyenne'] = df.loc[target_index, 'profit_net_monthly'] - market_avg_profit

    return df

def verifier_evenements_proches(city):
    """Charge les événements depuis events.json et vérifie si un événement majeur approche."""
    try:
        with open('events.json', 'r') as f:
            events_data = json.load(f)
    except FileNotFoundError:
        logger.warning("Fichier events.json non trouvé. Événements ignorés.")
        return False, None
    except json.JSONDecodeError:
        logger.error("Erreur de lecture du fichier events.json.")
        return False, None

    today = datetime.date.today()
    
    for event in events_data:
        if event.get("city").lower() == city.lower():
            try:
                start_date_obj = datetime.datetime.strptime(event["start_date"], "%Y-%m-%d").date()
            except ValueError:
                 logger.warning(f"Format de date invalide pour l'événement: {event['name']}. Ignoré.")
                 continue

            delta = start_date_obj - today
            
            if datetime.timedelta(days=0) <= delta <= datetime.timedelta(days=EVENT_PREDICTION_DAYS):
                logger.info(f"Événement majeur détecté: {event['name']} dans {delta.days} jours.")
                return True, event["name"]

    return False, None


def simuler_scenarios(df: pd.DataFrame, params: dict) -> dict:
    """Calcule l'impact sur le profit net en simulant différents ajustements de prix."""
    if df.empty or True not in df['is_target'].values:
        return {}

    target_data = df[df['is_target'] == True].iloc[0]
    
    prix_initial = target_data['price_per_night']
    occupation_initiale = target_data['occupancy']
    
    SCENARIOS = {
        "Ajustement Prudent (+5%)": {"price_delta_pct": 0.05, "occupancy_impact": -0.01},
        "Ajustement Fort (+15%)": {"price_delta_pct": 0.15, "occupancy_impact": -0.03},
        "Ajustement Agres. (+25%)": {"price_delta_pct": 0.25, "occupancy_impact": -0.05},
    }
    
    results = {}
    
    # Récupération des coûts personnalisés
    FRAIS_PLATEFORME_PCT = params.get('platform_fee_pct', 0.15)
    CHARGES_MENSUELLES_FIXES = params.get('monthly_fixed_costs', 300.0)
    CLEANING_FEE_PER_STAY = params.get('cleaning_fee_per_stay', 50.0)
    AVG_STAY_DURATION_NIGHTS = params.get('avg_stay_duration', 4)
    TAXE_DE_SEJOUR_PAR_NUIT = params.get('taxe_de_sejour_par_nuit', 1.50)
    
    for name, scenario_params in SCENARIOS.items():
        new_price = prix_initial * (1 + scenario_params["price_delta_pct"])
        new_occupancy = max(0, occupation_initiale + scenario_params["occupancy_impact"])
        
        new_revenue_monthly = (new_price * new_occupancy) * 30
        new_nuits_louees = new_occupancy * 30
        
        if AVG_STAY_DURATION_NIGHTS > 0:
            new_estim_sejours = new_nuits_louees / AVG_STAY_DURATION_NIGHTS
        else:
            new_estim_sejours = 0
            
        cost_platform = new_revenue_monthly * FRAIS_PLATEFORME_PCT
        cost_cleaning = new_estim_sejours * CLEANING_FEE_PER_STAY
        cost_taxes = new_nuits_louees * TAXE_DE_SEJOUR_PAR_NUIT
        
        new_profit = new_revenue_monthly - cost_platform - cost_cleaning - cost_taxes - CHARGES_MENSUELLES_FIXES
        
        results[name] = {
            "new_price": new_price,
            "new_occupancy": new_occupancy,
            "new_profit": new_profit,
            "profit_delta": new_profit - target_data['profit_net_monthly']
        }
        
    return results

# ----------------------------------------------------
# --- 💡 FONCTION MANQUANTE AJOUTÉE (API Google POI) ---
# ----------------------------------------------------

def fetch_dynamic_pois(client_lat, client_lon, city):
    """
    Appelle l'API Google Places pour obtenir les points d'intérêt (POI) 
    autour des coordonnées du client.
    """
    # ⚠️ REMPLACEZ "VOTRE_CLÉ_GOOGLE_PLACES_API" PAR VOTRE VRAIE CLÉ
    GOOGLE_PLACES_KEY = "AIzaSyCscmFRxZV7pOirVjXmGqyl3P-n0-iJVJM" 
    
    if GOOGLE_PLACES_KEY == "AIzaSyCscmFRxZV7pOirVjXmGqyl3P-n0-iJVJM":
        logger.warning("Clé Google Places manquante. Audit de proximité désactivé.")
        return []

    RADIUS_METERS = 800
    types_de_recherche = 'transit_station|tourist_attraction|lodging'
    
    url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
           f"location={client_lat},{client_lon}&radius={RADIUS_METERS}&type={types_de_recherche}&key={GOOGLE_PLACES_KEY}")
    
    try:
        response = requests.get(url, timeout=10) # Utilise requests (déjà importé)
        response.raise_for_status()
        data = response.json()
        
        return data.get('results', []) 
    
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Google Places API pour {city} : {e}")
        return []

# ----------------------------------------------------
# --- 💡 FONCTION MANQUANTE AJOUTÉE (Audit SEO) ---
# ----------------------------------------------------

def analyser_seo(client_data: dict, event_name=None):
    """
    Analyse le titre et la description, en incluant un
    audit de proximité GPS dynamique via Google Places API.
    """
    recommandations_seo = []
    
    title = client_data.get('current_title', "").lower()
    description = client_data.get('current_description', "").lower()
    city = client_data.get('city', "").lower()
    
    client_lat = client_data.get('target_latitude')
    client_lon = client_data.get('target_longitude')

    if not title: 
        return ["- ℹ️ Titre/Description non fournis. L'audit SEO est ignoré."]

    # --- Audit SEO de Base (inchangé) ---
    if len(title) < 40:
        recommandations_seo.append(
            "- **Titre (Visibilité)** : Votre titre est court. Un titre long et descriptif (ex: 'Superbe T2 - 5min Croisette') est mieux classé."
        )
    if event_name:
        if "festival" not in title and "congrès" not in title and "mipim" not in title:
             recommandations_seo.append(
                f"- **Titre (Événement)** : L'événement '{event_name}' approche. Mettez à jour votre titre pour inclure 'Spécial {event_name}' ou 'Proche Palais'."
            )
    if len(description) < 150:
        recommandations_seo.append(
            "- **Description (Conversion)** : Votre description est courte. Détaillez les équipements (WiFi, Parking, etc.)."
        )
    
    # --- Audit de Proximité DYNAMIQUE (GPS) ---
    
    PROXIMITY_THRESHOLD_KM = 0.8 # 800 mètres
    
    if client_lat and client_lon:
        
        pois_results = fetch_dynamic_pois(client_lat, client_lon, city)
        
        if pois_results:
            pois_found = 0
            for poi in pois_results:
                poi_lat = poi['geometry']['location']['lat']
                poi_lon = poi['geometry']['location']['lng']
                poi_name = poi['name']
                
                distance = calculate_distance(client_lat, client_lon, poi_lat, poi_lon)
                
                if distance <= PROXIMITY_THRESHOLD_KM:
                    keyword_to_use = next((k for k in ['Gare', 'Plage', 'Palais', 'Square', 'Park'] if k in poi_name), poi_name.split(' ')[0])

                    if keyword_to_use.lower() not in title:
                        recommandations_seo.append(
                            f"- **SEO (Proximité)** : Vous êtes à **{distance*1000:.0f}m** de '{poi_name}'. Ajoutez '{keyword_to_use}' à votre titre !"
                        )
                        pois_found += 1
            
            if pois_found == 0:
                 recommandations_seo.append(f"- ℹ️ Audit de Proximité : Aucun POI majeur trouvé dans un rayon de 800m.")
        else:
            recommandations_seo.append(f"- ℹ️ Audit de Proximité : Pas de POI trouvé ou API Google non configurée pour '{city}'.")
    else:
        recommandations_seo.append(f"- ℹ️ Audit de Proximité : Coordonnées GPS client manquantes.")

    return recommandations_seo

# ----------------------------------------------------
# --- 💡 FONCTION MANQUANTE AJOUTÉE (Audit Équipements) ---
# ----------------------------------------------------

def analyser_equipements(client_data: dict, competitors_df: pd.DataFrame):
    """
    Compare les équipements du client à ceux les plus mentionnés
    par les concurrents, en séparant les "Soft" (Actionnables) et "Hard" (Structurels).
    """
    if competitors_df.empty:
        return ["- ℹ️ Audit Équipements : Données concurrentielles insuffisantes."]
        
    client_amenities_raw = client_data.get('amenities_list') 
    
    if not client_amenities_raw: 
        client_amenities = set() 
    else:
        client_amenities = set([
            amenity.strip().lower() 
            for amenity in client_amenities_raw.split(',')
        ])
    
    if 'description' not in competitors_df.columns:
        logger.warning("Audit SEO : Colonne 'description' manquante dans le JSON des concurrents.")
        return ["- ℹ️ Audit Équipements : Descriptions concurrentes non trouvées."]
        
    all_descriptions = " ".join(competitors_df['description'].dropna().str.lower())
    
    recommendations_actionable = [] 
    recommendations_info = []       

    for keyword in SOFT_KEYWORDS: 
        if keyword in all_descriptions: 
            if keyword not in client_amenities:
                recommendations_actionable.append(
                    f"- **Équipement (Action)** : Vos concurrents mentionnent '{keyword}'. Si vous l'avez, **ajoutez-le à votre titre/description** pour plus de visibilité."
                )
    
    for keyword in HARD_KEYWORDS: 
        if keyword in all_descriptions: 
            if keyword not in client_amenities: 
                recommendations_info.append(
                    f"- **Marché (Info)** : Vos concurrents directs ont un avantage structurel ('{keyword}'). Cela peut justifier leur prix plus élevé (ou un prix plus bas du vôtre pour rester compétitif)."
                )
    
    final_report = []
    if recommendations_actionable:
        final_report.extend(recommendations_actionable)
    if recommendations_info:
        final_report.extend(recommendations_info)

    if not final_report:
        return ["- ✅ Équipements bien alignés sur le marché."]
        
    return final_report

# ----------------------------------------------------
# --- 💡 FONCTION MANQUANTE AJOUTÉE (Tendance) ---
# ----------------------------------------------------

def get_last_analysis_summary(client_id):
    """
    Récupère le prix moyen et le profit moyen de la DERNIÈRE analyse 
    effectuée pour ce client depuis l'historique (analysis_runs).
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 💡 CORRECTION : Utilise 'client_id' pour trouver le dernier run
        cursor.execute(
            "SELECT avg_price, avg_profit_net FROM analysis_runs WHERE client_id = ? ORDER BY date_run DESC LIMIT 1", 
            (client_id,)
        )
        last_run = cursor.fetchone()
        conn.close()
        
        if last_run:
            return dict(last_run) 
        else:
            return None # Retourne None si aucun historique n'est trouvé
            
    except Exception as e:
        logger.error(f"Impossible de lire l'historique d'analyse pour {client_id} : {e}")
        return None

# ----------------------------------------------------
# --- Fonctions de Reporting et Visualisation ---
# ----------------------------------------------------

def generer_recommandations(df):
    """Génère des recommandations ciblées pour la propriété de l'hôte, y compris les événements."""
    recommandations = []
    if df.empty or True not in df['is_target'].values: 
        return ["Aucune donnée disponible."]
    
    target_property = df[df['is_target'] == True].iloc[0]
    ecart_prix = target_property.get('ecart_prix_vs_moyenne', 0)
    
    event_proche, event_name = verifier_evenements_proches(target_property['city'])
    
    if event_proche:
        recommandations.append(f"🚨 ALERTE ÉVÉNEMENT MAJEUR ({event_name}) !")
        if ecart_prix < 50: 
            recommandations.append(
                f"- 🚀 **AUGMENTATION URGENTE** : Le congrès {event_name} maximise la demande. Augmentez vos prix de +20% à +50% pour la période concernée."
            )
        else:
            recommandations.append(
                "- ✅ Prix bien ajusté : Vous avez déjà anticipé la forte demande due à l'événement. Maintenez ce prix élevé."
            )
    else:
        if ecart_prix > 10: 
            recommandations.append(f"- 💸 Prix élevé (+{ecart_prix:.2f}€) : Votre prix est supérieur à la concurrence. Si l'occupation est faible, baissez légèrement le prix ou améliorez vos photos.")
        elif ecart_prix < -10: 
            recommandations.append(f"- 📈 PRIX SOUS-ESTIMÉ (-{abs(ecart_prix):.2f}€) : Vous sous-estimez votre bien. Vous pouvez augmenter immédiatement votre prix de {abs(ecart_prix):.2f}€.")
        else:
            recommandations.append("- ✅ Prix compétitif : Votre prix est aligné avec la concurrence directe. Concentrez-vous sur le service.")
            
    if target_property.get('ecart_profit_net_vs_moyenne', 0) < -50:
         recommandations.append("- 💰 Profit Net Faible : Votre profit est faible. Réduisez vos coûts fixes ou augmentez votre taux d'occupation.")

    if target_property.get('rating', 4.0) < NOTE_MIN_RECOMMANDATION:
        recommandations.append(f"- ⭐ Avis faibles ({target_property.get('rating', 0):.1f}) : Investissez dans la qualité du service pour justifier le prix.")

    return "\n" + "\n".join(recommandations)


def analyze_data(df, params: dict, client_data: dict, last_run_summary: dict = None):
    """
    Orchestre tous les calculs et agrège les résultats du rapport.
    Inclut maintenant l'analyse de tendance et l'audit SEO/Équipements.
    """
    
    if df.empty or True not in df['is_target'].values: 
        return "Aucune donnée analysée"
    
    target_property = df[df['is_target'] == True].iloc[0]
    competitor_df = df[df['is_target'] == False]
    
    scenarios = simuler_scenarios(df, params)
    
    # Construction du rapport principal
    rapport = [
        f"Analyse Stratégique pour **{target_property['name']}** à {target_property['city']} :",
        f"- Prix Actuel: **{target_property['price_per_night']:.2f} €/nuit**",
        f"- Prix Moyen des Concurrents: **{target_property.get('prix_moyen_concurrents', 0):.2f} €/nuit**",
        f"- Écart de Prix vs. Moyenne: **{target_property.get('ecart_prix_vs_moyenne', 0):.2f} €**",
    ]
    
    if last_run_summary:
        price_delta = target_property.get('prix_moyen_concurrents', 0) - last_run_summary.get('avg_price', 0)
        rapport.append(f"- **Tendance Marché (vs hier): {price_delta:+.2f} €**")
    
    rapport.append(f"- Profit Net Mensuel Estimé: **{target_property['profit_net_monthly']:.2f} €**")
    
    rapport.append("\n--- 💡 Simulation de Scénarios de Profit ---")
    for name, res in scenarios.items():
        delta_sign = '+' if res['profit_delta'] > 0 else ''
        rapport.append(
            f"- {name}: Prix Cible {res['new_price']:.2f}€ | Profit Estimé: {res['new_profit']:.0f}€ ({delta_sign}{res['profit_delta']:.0f}€) | Taux Occ.: {res['new_occupancy']*100:.1f}%"
        )
    rapport.append("---")

    event_proche, event_name = verifier_evenements_proches(target_property['city'])
    
    # Audit SEO (Titre/Description/GPS)
    recos_seo = analyser_seo(client_data, event_name if event_proche else None)
    if recos_seo:
        rapport.append("\n--- 🚀 Audit de Visibilité (SEO) ---")
        rapport.extend(recos_seo)

    # Audit des Équipements
    recos_equipements = analyser_equipements(client_data, competitor_df)
    if recos_equipements:
        rapport.append("\n--- ☕ Audit des Équipements (Marketing) ---")
        rapport.extend(recos_equipements)
        
    rapport.append("---") # Fin des sections SEO
    
    return "\n".join(rapport) + "\n\n" + generer_recommandations(df)


def generate_price_comparison_chart(df, output_dir):
    """Génère et sauvegarde le graphique de comparaison des prix (Fonctionnalité Startup)."""
    
    if df.empty or True not in df['is_target'].values: 
        logger.warning("Graphique non généré : aucune donnée cible.")
        return None
        
    target_data = df[df['is_target'] == True].iloc[0]
    
    prix_host = target_data['price_per_night']
    prix_moyen = target_data.get('prix_moyen_concurrents', prix_host)
    ecart_prix = target_data.get('ecart_prix_vs_moyenne', 0)
    profit_net_mensuel = target_data['profit_net_monthly']
    city = target_data['city']
    
    plt.figure(figsize=(8, 6))
    
    if ecart_prix < -10:
        color_host = '#28A745'
        alert_text = f"GAIN POTENTIEL : +{abs(ecart_prix):.2f} €/nuit"
        alert_color = '#28A745'
    elif ecart_prix > 10:
        color_host = '#C63333'
        alert_text = f"RISQUE : {ecart_prix:+.2f} €/nuit au-dessus de la cible"
        alert_color = '#C63333'
    else:
        color_host = '#1f77b4'
        alert_text = "ÉCART MINIME"
        alert_color = '#1f77b4'
        
    
    plt.bar(
        ["Mon Prix Actuel", "Moyenne Marché"], 
        [prix_host, prix_moyen], 
        color=[color_host, '#777777'], 
        alpha=0.9
    )
    
    event_proche, event_name = verifier_evenements_proches(city)
    
    if event_proche:
        main_title = f"🚨 ALERTE PRIX ÉVÉNEMENTIEL ({event_name})"
    else:
        main_title = f"Positionnement Tarifaire | {city}"
        
    plt.title(main_title, fontsize=16, weight='bold')
    
    plt.suptitle(
        f"{alert_text} | Profit Net Estimé: {profit_net_mensuel:.0f} €/mois", 
        y=0.95, 
        fontsize=10, 
        color=alert_color, 
        weight='bold'
    )
    
    plt.ylabel("Prix par Nuit (€)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.text(0, prix_host, f"{prix_host:.2f}€", ha='center', va='bottom', weight='bold')
    plt.text(1, prix_moyen, f"{prix_moyen:.2f}€", ha='center', va='bottom', weight='bold')
    
    chart_path = output_dir / f"price_comparison_{city}.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    
    logger.info(f"Graphique de comparaison de prix enregistré : {chart_path}")
    print(f"📈 Graphique de prix enregistré dans le dossier {OUTPUT_DIR.name}/")
    
    return chart_path


def generate_pdf_report(df_analyzed, chart_path, client_data, output_dir, params: dict, last_run_summary: dict = None):
    """Génère un rapport PDF professionnel (Platypus) intégrant le texte et le graphique."""
    
    if df_analyzed.empty or not chart_path or not os.path.exists(chart_path):
        logger.error("Impossible de générer le PDF : données ou graphique manquants.")
        return None

    target_data = df_analyzed[df_analyzed['is_target'] == True].iloc[0]
    
    # Génère le rapport textuel complet
    report_text = analyze_data(df_analyzed, params, client_data, last_run_summary)
    
    pdf_filename = output_dir / f"Rapport_Strategique_{client_data['client_name'].replace(' ', '_')}_{target_data['name'].replace(' ', '_')}.pdf"
    
    doc = SimpleDocTemplate(str(pdf_filename), pagesize=A4,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='NormalCode', fontName='Helvetica', fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=20, leading=26))
    styles.add(ParagraphStyle(name='H1', fontName='Helvetica-Bold', fontSize=14, leading=18))
    styles.add(ParagraphStyle(name='H2', fontName='Helvetica-Bold', fontSize=12, leading=15))
    
    Story = []
    
    # 1. EN-TÊTE ET VERDICT EXÉCUTIF
    ecart_prix = target_data.get('ecart_prix_vs_moyenne', 0)
    
    if ecart_prix < -20:
        verdict = f"📈 OPPORTUNITÉ DE PROFIT MAJEURE"
        verdict_color = colors.green
    elif ecart_prix > 20:
        verdict = f"❌ RISQUE DE NON-RÉSERVATION (PRIX ÉLEVÉ)"
        verdict_color = colors.red
    else:
        verdict = "✅ PRIX ALIGNÉ / TENDANCE STABLE"
        verdict_color = colors.blue
        
    Story.append(Paragraph(f"<b>Audit de Performance Tarifaire</b>", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph(f"<b>Statut Actuel :</b> {verdict}", ParagraphStyle('H1', textColor=verdict_color)))
    Story.append(Paragraph(f"<b>Propriété :</b> {target_data['name']} à {target_data['city']}", styles['Normal']))
    Story.append(Spacer(1, 0.4 * inch))

    # 2. INTÉGRATION DU GRAPHIQUE
    Story.append(Paragraph("<b>Section 1 : Positionnement Concurrentiel</b>", styles['H1']))
    Story.append(Spacer(1, 0.1 * inch))
    
    if Path(chart_path).exists():
        img = ImageReader(chart_path)
        img_width, img_height = img.getSize()
        width = 6 * inch 
        height = width * (img_height / img_width)
        Story.append(Image(chart_path, width=width, height=height))
        Story.append(Spacer(1, 0.2 * inch))
    
    # 3. ANALYSE DE SENSIBILITÉ (Extraction et formatage en tableau)
    scenarios_data = []
    is_scenario_block = False
    
    for line in report_text.split('\n'):
        if "Simulation de Scénarios de Profit" in line:
            is_scenario_block = True
            Story.append(Paragraph("<b>Section 2 : Simulation de Profit Net</b>", styles['H1']))
            Story.append(Spacer(1, 0.1 * inch))
            continue
        
        if is_scenario_block and line.startswith('- '):
            parts = line[2:].split(' | ')
            
            try:
                profit_str = [p for p in parts if 'Profit Estimé' in p][0].split(':')[1].split('(')[0].strip()
                profit_delta_str = re.search(r'\((.+?)\)', line)
                
                row_data = [
                    parts[0].split(':')[0].strip(), # Nom
                    [p for p in parts if 'Prix Cible' in p][0].split(':')[1].strip(), # Prix
                    [p for p in parts if 'Taux Occ.' in p][0].split(':')[1].strip(), # Occ
                    profit_str, # Profit
                    profit_delta_str.group(1).strip() if profit_delta_str else "0 €" # Delta
                ]
                scenarios_data.append(row_data)
            except IndexError:
                continue # Gérer les lignes mal formées
                
        if is_scenario_block and "---" in line and len(scenarios_data) > 0:
            is_scenario_block = False # Fin du bloc

    if scenarios_data:
        table_data = [
            ['Scénario', 'Prix Cible', 'Taux Occ.', 'Profit Net (€)', 'Delta (€)']
        ] + scenarios_data
        
        table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1.2*inch, 1*inch])
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        Story.append(table)
        Story.append(Spacer(1, 0.4 * inch))


    # 4. AUDIT SEO ET RECOMMANDATIONS (Conversion du reste du texte)
    Story.append(Paragraph("<b>Section 3 : Audit Marketing et Recommandations</b>", styles['H1']))
    Story.append(Spacer(1, 0.1 * inch))
    
    full_text_start_index = report_text.find("--- 🚀 Audit de Visibilité (SEO) ---") 
    
    if full_text_start_index != -1:
        text_for_pdf = report_text[full_text_start_index:].split('\n')
        
        for line in text_for_pdf:
            if line.strip() and "Simulation de Scénarios" not in line:
                line_html = line.replace('--- 🚀 Audit de Visibilité (SEO) ---', '<b>Audit de Visibilité (SEO)</b>')
                line_html = line_html.replace('--- ☕ Audit des Équipements (Marketing) ---', '<br/><b>Audit des Équipements (Marketing)</b>')
                line_html = line_html.replace('---', '')
                line_html = line_html.replace('- **', '<li><b>').replace('** :', '</b> :')
                line_html = line_html.replace('- 📈 ', '<li><b>')
                
                Story.append(Paragraph(line_html, styles['NormalCode']))

    # Construire le document
    try:
        doc.build(Story)
        logger.info(f"Rapport PDF généré : {pdf_filename}")
        print(f"📄 Rapport PDF généré : {pdf_filename}")
        return str(pdf_filename)
    except Exception as e:
        logger.error(f"Erreur fatale lors de la génération du PDF (Platypus) : {e}")
        return None

# --- Fonctions de Portefeuille (inchangées) ---
def generate_portfolio_pdf_report(client_name, portfolio_summary_df, output_dir):
    pdf_filename = output_dir / f"Rapport_Portefeuille_{client_name.replace(' ', '_')}.pdf"
    
    portfolio_summary_df = portfolio_summary_df.sort_values(by="profit_net_monthly", ascending=False)
    
    try:
        c = canvas.Canvas(str(pdf_filename), pagesize=A4)
        width, height = A4
        
        c.setFont('Helvetica-Bold', 18)
        c.drawString(inch, height - inch, f"Synthèse de Portefeuille - {client_name}")
        c.setFont('Helvetica', 10)
        c.drawString(inch, height - inch - 0.3 * inch, f"Date d'analyse : {datetime.date.today().strftime('%Y-%m-%d')}")
        c.line(inch, height - inch - 0.4 * inch, width - inch, height - inch - 0.4 * inch)
        
        y = height - inch - 0.8 * inch

        c.setFont('Helvetica-Bold', 10)
        c.drawString(inch, y, "Propriété")
        c.drawString(inch + 3 * inch, y, "Profit Net Estimé")
        c.drawString(inch + 4.5 * inch, y, "Écart Prix vs Marché")
        c.drawString(inch + 6 * inch, y, "Note")
        y -= 20

        c.setFont('Helvetica', 9)
        
        for _, row in portfolio_summary_df.iterrows():
            if y < inch: 
                c.showPage()
                y = height - inch
                c.setFont('Helvetica-Bold', 10)
                c.drawString(inch, y, "Propriété")
                c.drawString(inch + 3 * inch, y, "Profit Net Estimé")
                c.drawString(inch + 4.5 * inch, y, "Écart Prix vs Marché")
                c.drawString(inch + 6 * inch, y, "Note")
                y -= 20
                c.setFont('Helvetica', 9)

            c.drawString(inch, y, str(row['name']))
            c.drawString(inch + 3 * inch, y, f"{row['profit_net_monthly']:.2f} €")
            c.drawString(inch + 4.5 * inch, y, f"{row['ecart_prix_vs_moyenne']:.2f} €")
            c.drawString(inch + 6 * inch, y, str(row['rating']))
            y -= 14

        c.save()
        logger.info(f"Rapport de Portefeuille PDF généré : {pdf_filename}")
        print(f"📄 Rapport de Portefeuille PDF généré : {pdf_filename}")

    except Exception as e:
        logger.error(f"Erreur fatale lors de la génération du PDF Portefeuille: {e}")


# ----------------------------------------------------
# --- 💡 FONCTION DE COLLECTE (Lit le JSON local) ---
# ----------------------------------------------------

def load_real_data_from_json(client_data, max_listings):
    """
    Charge les données concurrentielles réelles depuis le fichier JSON 
    (airbnb_data.json) et mappe les champs réels, y compris le GPS.
    """
    city = client_data['city']
    logger.info(f"Début de l'analyse des données réelles depuis airbnb_data.json pour {city}.")
    print(f"✅ Démarrage du moteur d'analyse : Lecture du fichier JSON local...")
    
    data = []
    
    try:
        with open('airbnb_data.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
    except FileNotFoundError:
        logger.error("Fichier airbnb_data.json non trouvé ! Impossible de lire le marché.")
        print("❌ Fichier airbnb_data.json non trouvé ! Téléchargez-le depuis Apify.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logger.error("Erreur de lecture du fichier airbnb_data.json.")
        print("❌ Erreur de lecture du fichier JSON (format invalide).")
        return pd.DataFrame()

    # 💡 CORRECTION : Le JSON d'Apify est une LISTE
    if not isinstance(json_data, list):
        logger.warning(f"Le fichier JSON local n'a pas le format attendu (Liste). Trouvé : {type(json_data)}")
        print("❌ Le format JSON n'est pas une liste. Vérifiez le fichier téléchargé.")
        return pd.DataFrame()

    base_lat = client_data.get('target_latitude', 43.55) 
    base_lon = client_data.get('target_longitude', 7.01) 
        
    for item in json_data[:max_listings]:
        try:
            # 🎯 MAPPAGE JSON (Basé sur votre inspection)
            name = item.get('sharingConfigTitle', item.get('seoTitle', 'Titre inconnu')) 
            
            price = 0.0
            try:
                price_description = item.get('price', {}).get('breakDown', {}).get('basePrice', {}).get('description', "")
                price_match = re.search(r'[\$€]\s*([\d\.,]+)', price_description)
                if price_match:
                    price = float(price_match.group(1).replace(',', ''))
            except Exception as e:
                logger.warning(f"Impossible d'extraire le prix pour {name}: {e}")
                price = 0 

            size = item.get('personCapacity', 1) 
            
            rating_obj = item.get('rating', {})
            rating = rating_obj.get('value', 4.0) 
            if rating is None: rating = 4.0

            coords = item.get('coordinates', {})
            lat = coords.get('latitude')
            lon = coords.get('longitude')
            
            description = item.get('metaDescription', '') 

            data.append({
                "name": name,
                "city": city,
                "price_per_night": price,
                "rating": rating,
                "size": size, 
                "occupancy": random.uniform(0.7, 0.9), 
                "date_collected": time.strftime("%Y-%m-%d"),
                "is_target": False,
                "latitude": lat if lat else base_lat + random.uniform(-0.02, 0.02),
                "longitude": lon if lon else base_lon + random.uniform(-0.02, 0.02),
                "description": description 
            })
        except Exception as e:
            logger.warning(f"Impossible d'analyser un item JSON : {e}")
            continue
            
    logger.info(f"Analyse JSON terminée. {len(data)} concurrents trouvés.")
    print(f"✨ {len(data)} concurrents réels (via JSON) structurés en DataFrame.")
    
    return pd.DataFrame(data)

# ----------------------------------------------------
# --- Fonction : Récupérer les Clients de la DB ---
# ----------------------------------------------------

def get_all_clients_from_db():
    """Récupère tous les profils clients de la base de données."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clients")
        clients = cursor.fetchall()
        conn.close()
        return [dict(row) for row in clients]
    except Exception as e:
        logger.error(f"Impossible de récupérer les clients de la DB : {e}")
        return []

# ----------------------------------------------------
# --- 💡 PROGRAMME PRINCIPAL (MAIN) - Version SaaS GPS ---
# ----------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser(description="Bot d'analyse de listings Airbnb.")
    parser.add_argument('-n', "--num-listings", type=int, default=DEFAULT_LISTINGS, help=f"Listings max (Défaut: {DEFAULT_LISTINGS}).")
    parser.add_argument('--skip-export', action='store_true', help="Ignorer l'export Google Sheets.")
    
    args = parser.parse_args()
    setup_database() 
    
    print("\n--- Démarrage du Bot Airbnb Automatique (Mode SaaS - JSON Local) ---")
    
    clients_to_analyze = get_all_clients_from_db()

    if not clients_to_analyze:
        print("ℹ️ Aucun client à analyser. Exécutez 'python3 add_client.py' pour commencer.")
        return

    print(f"✅ {len(clients_to_analyze)} client(s) trouvé(s). Démarrage de l'analyse en boucle...")
    
    clients_df = pd.DataFrame(clients_to_analyze)
    grouped_clients = clients_df.groupby('client_name')
    
    for client_name, properties_df in grouped_clients:
        
        print("\n" + "="*50)
        logger.info(f"Traitement du Portefeuille Client : {client_name}")
        
        all_analyses_for_this_client = [] 

        for _, client_row in properties_df.iterrows():
            
            client_data = dict(client_row) 
            client_city = client_data['city']
            property_name = client_data['target_property_name']
            
            print(f"\n--- Analyse de la Propriété : {property_name} ({client_city}) ---")
            
            # 2. Collecte des données
            df_competitors = load_real_data_from_json(client_data, args.num_listings) 
            
            if df_competitors.empty:
                logger.error(f"Échec de la lecture du JSON pour {client_city}. Propriété ignorée.")
                print(f"🛑 Arrêt de l'analyse pour {property_name} (Échec de la lecture JSON).")
                continue

            # 2.5. Préparer la propriété cible
            df_target = pd.DataFrame([{
                "name": client_data['target_property_name'],
                "city": client_data['city'],
                "price_per_night": client_data['target_price'],
                "rating": client_data.get('target_rating', 4.0),
                "size": client_data.get('target_size', 50),
                "occupancy": client_data.get('target_occupancy', 0.7),
                "date_collected": time.strftime("%Y-%m-%d"),
                "is_target": True,
                "latitude": client_data.get('target_latitude'),
                "longitude": client_data.get('target_longitude'),
                "description": client_data.get('current_description') 
            }])
            
            df_listings = pd.concat([df_target, df_competitors], ignore_index=True)
            
            # 3. Chaîne d'Analyse Complète
            df_listings = calculer_revenus(df_listings) 
            df_listings = calculer_profit_net(df_listings, client_data) 
            df_listings = analyser_concurrentiel(df_listings) # Utilise maintenant le GPS
            
            # 💡 CORRECTION : Passe le client_id pour la recherche de tendance
            last_run_summary = get_last_analysis_summary(client_data['client_id'])
            
            analysis_output = analyze_data(df_listings, client_data, client_data, last_run_summary)
            chart_path = generate_price_comparison_chart(df_listings, OUTPUT_DIR) 

            # 4. Génération du Rapport PDF (par propriété)
            pdf_path = None 
            if chart_path: 
                try:
                    pdf_path = generate_pdf_report(df_listings, chart_path, client_data, OUTPUT_DIR, client_data, last_run_summary) 
                except Exception as e:
                    logger.error(f"Erreur FATALE lors de la création du PDF pour {client_name}: {e}")
                    print(f"❌ Erreur lors de la création du PDF pour {client_name}. L'e-mail sera envoyé sans pièce jointe.")
            else:
                logger.error(f"Génération PDF annulée pour {client_name}, graphique manquant.")

            # ----------------------------------------------------
            # --- 📧 BLOC D'ALERTE EMAIL INTÉGRÉ ---
            # ----------------------------------------------------
            
            target_property_df = df_listings[df_listings['is_target'] == True]
            current_ecart_prix = target_property_df['ecart_prix_vs_moyenne'].iloc[0]
            client_email = client_data.get('client_email')
            
            event_proche, event_name = verifier_evenements_proches(client_city)
            
            alert_needed = False
            alert_subject = ""
            email_body = ""

            if client_email: 
                if event_proche:
                    alert_needed = True
                    alert_subject = f"🚨 ALERTE ÉVÉNEMENT : Hausse de prix urgente pour {property_name} ({event_name})!"
                    email_body = (
                        f"Bonjour {client_name},\n\nNotre Bot SaaS a détecté que l'événement '{event_name}' approche.\n"
                        f"C'est une opportunité d'optimisation de prix urgente.\n\n"
                        "Veuillez consulter le rapport PDF ci-joint pour les détails.\n\n"
                        "Cordialement,\nVotre Partenaire Analyse."
                    )
                
                elif current_ecart_prix < -20: 
                    alert_needed = True
                    alert_subject = f"📈 OPPORTUNITÉ : Votre bien {property_name} est sous-estimé de {abs(current_ecart_prix):.0f}€ !"
                    email_body = (
                        f"Bonjour {client_name},\n\nNotre analyse a détecté que votre prix est {abs(current_ecart_prix):.0f}€ en dessous de la moyenne du marché comparable.\n"
                        "Vous laissez de l'argent sur la table.\n\n"
                        "Veuillez consulter le rapport PDF ci-joint pour les scénarios d'augmentation de profit.\n\n"
                        "Cordialement,\nVotre Partenaire Analyse."
                    )

                if alert_needed:
                    try:
                        send_email_alert(
                            client_email,
                            alert_subject,
                            email_body,
                            attachment_path=pdf_path if pdf_path and os.path.exists(pdf_path) else None 
                        )
                        print(f"📧 Alerte email envoyée avec succès à {client_email}")
                    except Exception as e:
                        logger.error(f"Échec critique de l'envoi d'e-mail pour {client_name} : {e}")
                        print(f"❌ Échec de l'envoi de l'e-mail pour {client_name}. (Non bloquant pour l'analyse).")
            
            # --- FIN DU BLOC D'ALERTE ---

            print("\n" + "="*30)
            print(f"🤖 Rapport (Propriété) pour {property_name} :")
            print(analysis_output)
            print("="*30)

            # 5. Sauvegarde et Export
            save_to_sqlite(df_listings, client_data['client_id'], client_city) # 💡 CORRECTION : Passe le client_id ET la ville
            
            if not args.skip_export:
                save_to_gsheet(df_listings, f"{client_name}_{property_name}") 

            all_analyses_for_this_client.append(df_listings[df_listings['is_target'] == True])

        # --- Fin de la boucle des propriétés ---
        
        # 6. Générer le rapport de portefeuille pour ce client
        if all_analyses_for_this_client:
            portfolio_summary_df = pd.concat(all_analyses_for_this_client)
            generate_portfolio_pdf_report(client_name, portfolio_summary_df, OUTPUT_DIR)
        
    # --- Fin de la boucle des clients ---
    
    logger.info("--- Bot Airbnb terminé avec succès (Tous les clients traités) ---")
    print("\n✅ Tâches terminées pour tous les clients. Consultez 'bot_airbnb.log' pour le suivi.")
    print("➜ bot airbnb") 

if __name__ == '__main__':
    main()