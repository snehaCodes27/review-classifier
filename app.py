from flask import Flask, request, send_file, render_template
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download once (safe if already installed)
nltk.download("vader_lexicon")

app = Flask(__name__)

OUTPUT_FILE = "output.csv"

@app.route("/", methods=["GET", "POST"])
def home():
    msg = ""

    if request.method == "POST":

        if "file" not in request.files:
            msg = "No file uploaded"
            return render_template("home.html", msg=msg)

        file = request.files["file"]

        if file.filename == "":
            msg = "Please select a file"
            return render_template("home.html", msg=msg)

        data = file.read().decode("utf-8")
        lines = data.splitlines()[1:]  # skip header

        sia = SentimentIntensityAnalyzer()

        result = "reviews,score,label\n"

        for line in lines:
            text = line.strip()
            ps = sia.polarity_scores(text)

            if ps["compound"] >= 0.05:
                label = "Positive"
            elif ps["compound"] <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"

            result += f"{text},{ps['compound']},{label}\n"

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(result)

        return send_file(OUTPUT_FILE, as_attachment=True)

    return render_template("home.html", msg=msg)


if __name__ == "__main__":
    pass   # teacher requirement
