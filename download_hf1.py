from datasets import load_dataset, Audio
import os

os.makedirs("data/authentic", exist_ok=True)
os.makedirs("data/deepfakes", exist_ok=True)

print("Connexion au dataset Hugging Face (Mode Streaming)...")
dataset = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True)

# L'ASTUCE EST ICI : On désactive le décodage pour contourner torchcodec !
dataset = dataset.cast_column("audio", Audio(decode=False))

# Définition des objectifs exacts
TARGET_REAL = 200
TARGET_FAKE = 200 

real_count = 0
fake_count = 0

print(f"Extraction ciblée : {TARGET_REAL} audios réels et {TARGET_FAKE} deepfakes en cours...")

for row in dataset:
    # Condition d'arrêt absolu : on arrête quand les DEUX dossiers ont atteint 200
    if real_count >= TARGET_REAL and fake_count >= TARGET_FAKE:
        break
        
    system_id = row.get("system_id", "")
    
    # Cas 1 : C'est une voix authentique
    if system_id == "-":
        if real_count < TARGET_REAL:
            filepath = f"data/authentic/asvspoof_real_{real_count:04d}.wav"
            with open(filepath, "wb") as f:
                f.write(row["audio"]["bytes"])
            real_count += 1
            print(f"Progression -> Réels: {real_count}/{TARGET_REAL} | Fakes: {fake_count}/{TARGET_FAKE}", end="\r")
            
    # Cas 2 : C'est une voix générée par IA (deepfake)
    else:
        if fake_count < TARGET_FAKE:
            filepath = f"data/deepfakes/asvspoof_fake_{fake_count:04d}.wav"
            with open(filepath, "wb") as f:
                f.write(row["audio"]["bytes"])
            fake_count += 1
            print(f"Progression -> Réels: {real_count}/{TARGET_REAL} | Fakes: {fake_count}/{TARGET_FAKE}", end="\r")

print(f"\n✅ Terminé ! {real_count} audios réels et {fake_count} deepfakes ont été sauvegardés avec succès.")