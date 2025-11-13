# email_client.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os

# --- Configuration Email (À adapter) ---
SMTP_SERVER = 'smtp.gmail.com' # Exemple pour Gmail
SMTP_PORT = 587
EMAIL_ADDRESS = 'yassinemaalla0@gmail.com' # ⬅️ REMPLACEZ PAR VOTRE EMAIL
EMAIL_PASSWORD = 'mfkhhbstprmuascu' # ⬅️ REMPLACEZ PAR LE MOT DE PASSE D'APPLICATION

# email_client.py

def send_email_alert(recipient_email, subject, body, attachment_path=None):
    """Envoie un email avec ou sans pièce jointe (Rapport PDF)."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(attach)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        
        # 💡 CORRECTION : Message de succès décommenté
        print(f"📧 Alerte email envoyée avec succès à {recipient_email}")
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de l'email à {recipient_email}: {e}")
        return False

# --- Exemple d'utilisation (pour test) ---
# Dans email_client.py, si vous le lancez directement
if __name__ == '__main__':
    print("Test d'envoi d'alerte :")
    send_email_alert(
        recipient_email="yassinemaalla0@gmail.com", # ⬅️ Mettez votre propre email ici
        subject="[TEST] Configuration du Bot Réussie",
        body="Connexion Gmail OK.",
        attachment_path=None
    )