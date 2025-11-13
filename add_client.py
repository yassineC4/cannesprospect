# add_client.py (Version Finale Robuste)

import sqlite3
import sys
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

DB_NAME = "airbnb_history.db"

def geocode_address(address, city): # 💡 AJOUT DE 'city'
    """Convertit une adresse en latitude et longitude en utilisant Nominatim."""
    try:
        geolocator = Nominatim(user_agent="proptech_analyzer")
        
        # 💡 CORRECTION : Force la recherche dans la bonne ville
        location = geolocator.geocode(f"{address}, {city}", timeout=10) 
        
        if location:
            return location.latitude, location.longitude
        print("⚠️ Adresse non trouvée. Essai sans la ville...")
        # Essai n°2 (sans la ville, si la première échoue)
        location_fallback = geolocator.geocode(address, timeout=10)
        if location_fallback:
            return location_fallback.latitude, location_fallback.longitude
            
        print("❌ Adresse non trouvée par l'API de géocodage.")
        return None, None
    except Exception as e:
        print(f"❌ Erreur de géocodage: {e}")
        return None, None

def add_new_client():
    """Script simple pour ajouter un nouveau client à la base de données."""
    try:
        print("--- Ajout d'un nouveau client (Version GPS) ---")
        
        client_name = input("Nom du client : ")
        property_name = input("Nom de la propriété : ")
        city = input("Ville : ") # 💡 On demande la ville AVANT l'adresse
        
        full_address = input(f"Adresse COMPLÈTE à {city} (Rue, Code Postal) : ")
        
        latitude, longitude = geocode_address(full_address, city) # 💡 On passe la ville
        
        if latitude is None:
            print(f"❌ Impossible de localiser '{full_address}'. Client non ajouté.")
            return

        print(f"✅ Coordonnées trouvées : Lat={latitude:.4f}, Lng={longitude:.4f}")
        
        # ... (Le reste du script est identique : prix, taille, SEO, email, insertion DB) ...
        
        print("\n--- Paramètres Financiers et SEO ---")
        price = float(input("Prix actuel par nuit : "))
        size = float(input("Superficie (m²) : "))
        rating = float(input("Note actuelle : "))
        occupancy = float(input("Taux d'occupation : "))
        fixed_costs = float(input("Coûts fixes mensuels : "))
        current_title = input("Quel est le TITRE ACTUEL de votre annonce ? : ")
        current_description = input("Copiez-collez la DESCRIPTION ACTUELLE : ")
        client_email = input("E-mail du client (pour les alertes) : ") 
        amenities_list = input("Listez vos équipements clés (ex: WiFi,Nespresso,Clim,Parking) : ")

        conn = sqlite3.connect(DB_NAME)
        
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
        
        conn.execute("""
            INSERT INTO clients (client_name, target_property_name, city, target_price, 
                                 target_size, target_rating, target_occupancy, monthly_fixed_costs, 
                                 current_title, current_description, client_email, 
                                 target_latitude, target_longitude,
                                 cleaning_fee_per_stay, avg_stay_duration, taxe_de_sejour_par_nuit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (client_name, property_name, city, price, 
              size, rating, occupancy, fixed_costs, 
              current_title, current_description, client_email, 
              latitude, longitude,
              50.0, 4, 1.5)) # Exemple de valeurs par défaut pour les nouveaux champs
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Client '{client_name}' (Email: {client_email}) ajouté avec succès (Lat: {latitude:.4f}) !")

    except ValueError:
        print("❌ Erreur : Entrée numérique invalide. Veuillez réessayer.")
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout du client : {e}")

if __name__ == "__main__":
    add_new_client()