# guest_estimate.py
import streamlit as st
import pandas as pd
import random
import json
from datetime import datetime
import math

# --- Constantes (Synchronisation) ---
# Ces valeurs sont utilisées pour le calcul simulé des coûts
FRAIS_PLATEFORME_PCT = 0.15 
FRAIS_NETTOYAGE_PCT = 0.10   
CHARGES_MENSUELLES_FIXES = 350.0 # Coûts moyens estimés pour une estimation gratuite


def simuler_concurrentiel(city):
    """Simule les données du marché (comme dans main_logic.py) pour la cohérence."""
    # Le prix moyen est basé sur un seed pour la même date, mais sans historique.
    random.seed(datetime.now().date().strftime('%Y%m%d') + city.lower())
    
    avg_price = random.uniform(100.0, 150.0)
    avg_note = random.uniform(4.0, 4.8)
    
    # Simulation de l'impact événementiel (si events.json existe)
    try:
        with open('events.json', 'r') as f:
            events = json.load(f)
            if any(e.get('city').lower() == city.lower() for e in events):
                # Ajoute un bonus si un événement est proche
                avg_price *= 1.10 
    except:
        pass # Pas de fichier events, pas de bonus

    return round(avg_price, 2), round(avg_note, 1)

def calculer_profit_net_gratuit(avg_price, occupancy):
    """Calcul du profit net pour l'estimation gratuite (basé sur des hypothèses générales)."""
    revenue_monthly = (avg_price * occupancy) * 30
    cost_total = revenue_monthly * (FRAIS_PLATEFORME_PCT + FRAIS_NETTOYAGE_PCT) + CHARGES_MENSUELLES_FIXES
    profit_net = revenue_monthly - cost_total
    return profit_net

# --- Streamlit UI pour l'Estimation Gratuite ---

st.set_page_config(
    page_title="🚀 Estimation Gratuite Airbnb",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("💰 Obtenez votre Estimation de Revenu Airbnb Gratuite")
st.markdown("Entrez simplement le nom de votre ville pour connaître le potentiel de profit de votre marché.")

city_input = st.text_input("Ville (Ex: Cannes, Paris, Lyon)", placeholder="Entrez le nom de votre ville")

if city_input:
    city = city_input.strip()
    
    with st.spinner(f"Analyse du marché à {city}..."):
        avg_market_price, avg_market_note = simuler_concurrentiel(city)

        # Hypothèses générales de performance pour un bien standard
        occupancy_standard = 0.70 
        
        # Calcul du profit pour 3 scénarios de prix
        profit_base = calculer_profit_net_gratuit(avg_market_price, occupancy_standard)
        profit_optimiste = calculer_profit_net_gratuit(avg_market_price * 1.10, occupancy_standard * 0.95) # 10% plus cher, mais l'occupation baisse légèrement

    st.markdown("---")
    st.header(f"Résultats de l'Analyse du Marché pour {city}")
    
    col_kpi, col_details = st.columns([1, 2])

    with col_kpi:
        st.metric("Prix Moyen de Marché Estimé", f"{avg_market_price:.2f} €/nuit")
        st.metric("Note Moyenne des Concurrents", f"{avg_market_note:.1f} / 5.0")
        st.metric("Potentiel de Revenue Brut", f"{avg_market_price * 30 * occupancy_standard * 1.2 :.0f} €/mois")
        
    with col_details:
        st.subheader("Potentiel de Profit Net Mensuel (Estimations Basées sur Coûts Moyens)")
        
        data = {
            "Scénario": [
                "Prix Standard (70% Occ.)", 
                "Prix Optimisé (+10%)",
                "Besoin d'un Rapport Détaillé ?"
            ],
            "Prix Cible (€/nuit)": [
                avg_market_price, 
                avg_market_price * 1.10, 
                "..."
            ],
            "Profit Net Estimé (€/mois)": [
                f"**{profit_base:.0f} €**", 
                f"**{profit_optimiste:.0f} €**",
                "..."
            ]
        }
        df_estimate = pd.DataFrame(data)
        st.dataframe(df_estimate, hide_index=True, use_container_width=True)
        
        st.success(f"**Conclusion :** Votre marché à {city} a un potentiel de profit net entre {profit_base:.0f} € et {profit_optimiste:.0f} € par mois (avec un bien bien géré).")
    
    st.markdown("---")
    st.subheader("Passez à l'action avec notre Bot Complet")
    st.warning("Ceci n'est qu'une estimation. Pour un calcul précis de votre profit (avec vos coûts et votre adresse exacte) et des recommandations de prix dynamique, commandez un rapport complet.", icon="📈")