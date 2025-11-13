import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
import datetime
import json
import math

# --- Configuration et Constantes ---
# Ces valeurs sont utilisées comme DÉFAUT si la base de données n'est pas complète
DB_NAME = "airbnb_history.db"
OUTPUT_DIR = Path("output")

# Constantes pour la logique d'analyse (Copie des valeurs par défaut)
FRAIS_PLATEFORME_PCT = 0.15 
FRAIS_NETTOYAGE_PCT = 0.10   
CHARGES_MENSUELLES_FIXES = 300.0 
NOTE_MIN_RECOMMANDATION = 4.0 
EVENT_PREDICTION_DAYS = 60 

# ----------------------------------------------------------------------
# 1. Fonctions d'Analyse (Logique Métier Autonome)
# ----------------------------------------------------------------------

def verifier_evenements_proches(city):
    """Charge les événements depuis events.json et vérifie si un événement majeur approche."""
    try:
        with open('events.json', 'r') as f:
            events_data = json.load(f)
    except FileNotFoundError:
        return False, None
    except json.JSONDecodeError:
        return False, None

    today = datetime.date.today()
    
    for event in events_data:
        if event.get("city").lower() == city.lower():
            try:
                start_date_obj = datetime.datetime.strptime(event["start_date"], "%Y-%m-%d").date()
            except ValueError:
                 continue

            delta = start_date_obj - today
            
            if datetime.timedelta(days=0) <= delta <= datetime.timedelta(days=EVENT_PREDICTION_DAYS):
                return True, event["name"]

    return False, None


def get_event_alert(city):
    """Vérifie et retourne l'état de l'alerte événementielle."""
    return verifier_evenements_proches(city)

# 💡 CORRECTION (UnboundLocalError) : 
# La fonction utilise maintenant les constantes globales de app.py
def calculer_profit_net(df, params: dict):
    """Calcule le profit net mensuel estimé en utilisant les coûts personnalisés."""
    if df.empty: return df
    
    # Utilise les paramètres s'ils sont fournis, sinon les constantes globales de app.py
    _platform_pct = params.get('platform_fee_pct', FRAIS_PLATEFORME_PCT)
    _cleaning_pct = params.get('cleaning_fee_pct', FRAIS_NETTOYAGE_PCT)
    _fixed_costs = params.get('monthly_fixed_costs', CHARGES_MENSUELLES_FIXES)
    
    df['cost_total'] = df['revenue_monthly'] * (_platform_pct + _cleaning_pct) + _fixed_costs
    df['profit_net_monthly'] = df['revenue_monthly'] - df['cost_total']
    
    return df

# 💡 CORRECTION (TypeError) :
# La fonction accepte bien 'params' (un dict vide) et utilise les données de 'latest_run'
# app.py (Remplacez cette fonction)

def simuler_scenarios(latest_run_data, params: dict):
    """Calcule l'impact sur le profit net en simulant différents ajustements de prix."""
    
    # Base sur le prix moyen enregistré de la dernière exécution
    prix_initial = latest_run_data['avg_price'] 
    
    # Hypothèse d'occupation par défaut
    occupation_initiale = 0.75 

    SCENARIOS = {
        "Actuel (Base Marché)": {"price_delta_pct": 0.00, "occupancy_impact": 0.00},
        "Prudent (+5%)": {"price_delta_pct": 0.05, "occupancy_impact": -0.01}, 
        "Fort (+15%)": {"price_delta_pct": 0.15, "occupancy_impact": -0.03},  
        "Agres. (+25%)": {"price_delta_pct": 0.25, "occupancy_impact": -0.05},
    }
    
    results = []
    
    # Calcul du profit de base pour la comparaison delta
    base_revenue = (prix_initial * (occupation_initiale + 0.00)) * 30
    temp_df_base = pd.DataFrame([{'revenue_monthly': base_revenue}])
    
    # 💡 On utilise 'params' (les coûts) pour le calcul de base
    base_profit = calculer_profit_net(temp_df_base, params).iloc[0]['profit_net_monthly']
    
    results.append({
        "Scénario": "Actuel (Base Marché)",
        "Prix Cible (€)": round(prix_initial, 2),
        "Taux Occ. (%)": round(occupation_initiale * 100, 1),
        "Profit Net (€)": round(base_profit, 0),
        "Delta vs Actuel (€)": 0
    })

    # Calcul des autres scénarios
    # 💡 CORRECTION : Renommage de 'params' en 's_params' pour éviter le conflit
    for name, s_params in SCENARIOS.items():
        if name == "Actuel (Base Marché)":
            continue # Déjà fait

        new_price = prix_initial * (1 + s_params["price_delta_pct"])
        new_occupancy = max(0, occupation_initiale + s_params["occupancy_impact"])
        new_monthly_revenue = (new_price * new_occupancy) * 30
        
        temp_df = pd.DataFrame([{'revenue_monthly': new_monthly_revenue}])
        
        # 💡 CORRECTION : Utilise 'params' (les coûts), et non 's_params'
        profit_net_df = calculer_profit_net(temp_df, params) 
        new_profit = profit_net_df.iloc[0]['profit_net_monthly']
        
        profit_delta = new_profit - base_profit
        
        results.append({
            "Scénario": name,
            "Prix Cible (€)": round(new_price, 2),
            "Taux Occ. (%)": round(new_occupancy * 100, 1),
            "Profit Net (€)": round(new_profit, 0),
            "Delta vs Actuel (€)": round(profit_delta, 0)
        })
    
    return results


# ----------------------------------------------------------------------
# 2. Fonction de Chargement et de Rendu (UI)
# ----------------------------------------------------------------------

@st.cache_data
def load_data_from_db():
    """Charge les données de l'historique depuis SQLite."""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = "SELECT * FROM analysis_runs ORDER BY date_run DESC LIMIT 10"
        df = pd.read_sql_query(query, conn, index_col=None)
        conn.close()
        return df
    except sqlite3.Error as e:
        # st.error(f"Erreur de lecture de la base de données : {e}")
        # Retourne un DF vide si la DB n'existe pas
        return pd.DataFrame()


# --- Démarrage de l'Application Streamlit ---

st.set_page_config(
    page_title="Dashboard Stratégique Airbnb",
    layout="wide",
    initial_sidebar_state="expanded"
)

df_history = load_data_from_db()

# --- Affichage du Dashboard ---

if df_history.empty:
    st.title("🚀 Tableau de Bord Stratégique de Tarification")
    st_icon = "ℹ️"
    st.info("Aucune donnée d'analyse trouvée dans la base de données. Veuillez exécuter `python3 main_logic.py` au moins une fois pour générer un rapport.", icon=st_icon)
else:
    # Récupérer les données de la dernière exécution
    latest_run = df_history.iloc[0]
    city_name = latest_run['city']
    
    # Déterminer la Tendance et l'Alerte
    if len(df_history) > 1:
        profit_delta_hist = latest_run['avg_profit_net'] - df_history.iloc[1]['avg_profit_net']
        price_delta_hist = latest_run['avg_price'] - df_history.iloc[1]['avg_price']
    else:
        profit_delta_hist = 0
        price_delta_hist = 0
        
    # L'hypothèse de coûts est {} car nous utilisons les constantes par défaut de app.py
    cost_params = {} 
    event_proche, event_name = get_event_alert(city_name)

    # --- TITRE PRINCIPAL ---
    st.title(f"🚀 Tableau de Bord Stratégique | {city_name}")
    if event_proche:
        st.error(f"🚨 ALERTE URGENTE : Événement Majeur ({event_name}) détecté ! Ajustement des prix nécessaire.", icon="🚨")
    
    st.markdown("---")

    # 1. Vue d'ensemble (KPIs)
    st.header("Analyse de Tendance et Métriques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Prix Moyen Marché Actuel", f"{latest_run['avg_price']:.2f} €", delta=f"{price_delta_hist:+.2f} € vs. Préc.")
    col2.metric("Profit Net Moyen Estimé", f"{latest_run['avg_profit_net']:.0f} €", delta=f"{profit_delta_hist:+.0f} € vs. Préc.")
    col3.metric("Nombre d'Annonces Suivies", latest_run['num_listings'])
    col4.metric("Date de la Dernière Analyse", latest_run['date_run'].split()[0])

    st.markdown("---")

    # 2. Visualisation du Graphique et Simulation de Scénarios
    col_chart, col_scenarios = st.columns([6, 4])
    
    with col_chart:
        st.subheader("Positionnement Tarifaire Actuel (vs Moyenne Marché)")
        # Affichage de l'image PNG (générée par main_logic.py)
        chart_path = OUTPUT_DIR / f"price_comparison_{city_name}.png"
        if os.path.exists(chart_path):
            st.image(str(chart_path), caption="Comparaison de Votre Prix avec la Moyenne du Marché")
        else:
            st.warning("Graphique de comparaison non trouvé. Exécutez `main_logic.py`.")

    with col_scenarios:
        st.subheader("Analyse de Sensibilité (Profit Net)")
        
        # 💡 CORRECTION : Appelle la simulation avec les bons arguments
        scenarios_df = pd.DataFrame(simuler_scenarios(latest_run, cost_params)) 
        
        # Afficher le tableau des scénarios
        st.dataframe(scenarios_df, width='stretch', hide_index=True)
        st.caption("Montre l'impact de l'augmentation du prix sur le Profit Net estimé.")
    
    st.markdown("---")

    # 3. Historique des Exécutions (La preuve que le bot travaille)
    st.subheader("Historique des Performances (Preuve de la Fiabilité)")
    
    df_display = df_history[['date_run', 'city', 'num_listings', 'avg_price', 'avg_profit_net']].copy()
    df_display.columns = ['Date', 'Ville', 'Listings', 'Prix Moyen Marché', 'Profit Net Moyen']
    # 💡 CORRECTION (Avertissement de dépréciation) : Remplacement de use_container_width
    st.dataframe(df_display, width='stretch', hide_index=True)