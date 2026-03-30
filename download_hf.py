from datasets import load_dataset, Audio
import os

os.makedirs("data/authentic", exist_ok=True)
os.makedirs("data/deepfakes", exist_ok=True)

print("Connexion au dataset Hugging Face (Mode Streaming)...")
dataset = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True)

# L'ASTUCE EST ICI : On désactive le décodage pour contourner torchcodec !
dataset = dataset.cast_column("audio", Audio(decode=False))

LIMIT = 1000
count = 0

print(f"Extraction de {LIMIT} fichiers audio purs en cours...")

for row in dataset:
    if count >= LIMIT:
        break
        
    system_id = row.get("system_id", "")
    
    if system_id == "-":
        filepath = f"data/authentic/asvspoof_real_{count:04d}.wav"
    else:
        filepath = f"data/deepfakes/asvspoof_fake_{count:04d}.wav"
        
    # On écrit directement les octets (bytes) dans le fichier
    with open(filepath, "wb") as f:
        f.write(row["audio"]["bytes"])
        
    count += 1

print(f"✅ Terminé ! {count} fichiers ont été sauvegardés.")