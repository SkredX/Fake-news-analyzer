import pickle
import os

# Create a folder to save model and vectorizer
save_dir = r"D:\FastAPI\Fake_News_Project\model"
os.makedirs(save_dir, exist_ok=True)

# Save model
with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")
