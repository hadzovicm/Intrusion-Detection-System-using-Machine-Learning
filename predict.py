import pickle

# Load saved objects
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def classify_request(request):
    request = request.lower()
    vector = tfidf.transform([request])
    prediction = model.predict(vector)[0]
    return prediction

# Example usage
while True:
    req = input("Enter request: ")
    print("Malicious" if classify_request(req) == 1 else "Normal")
