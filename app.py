from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "model_trained_101class.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FCv3UeEIfw4Qs_A9U74YjbT85XeNrUyr"


def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found. Downloading from Google Drive using gdown...")
        try:
            subprocess.run([
                "python", "-m", "gdown",
                f"https://drive.google.com/uc?id=1FCv3UeEIfw4Qs_A9U74YjbT85XeNrUyr",
                "-O", MODEL_PATH
            ], check=True)
            print("Model downloaded successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download model file with gdown. Error: {e}")




labels = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
    "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella",
    "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
    "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

def get_macros(label):
    return {
        "food": label,
        "calories": 250,
        "protein_kcal": 20,
        "carbs_kcal": 150,
        "fat_kcal": 80
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    download_model_if_missing()
    model = keras.models.load_model(MODEL_PATH)
    image = Image.open(io.BytesIO(await file.read())).resize((299, 299))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)
    label = labels[np.argmax(preds[0])]
    return get_macros(label)
