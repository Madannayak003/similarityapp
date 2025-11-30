from flask import Flask, render_template, request, send_from_directory
import os
from processor import process_documents

app = Flask(__name__)

# Render-safe upload folder
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SERVE FILES FROM uploads FOLDER
@app.route("/uploads/<path:filename>")
def uploaded_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# DOWNLOAD FILES FROM uploads FOLDER
@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    uploaded_files = request.files.getlist("files")

    file_paths = []
    for file in uploaded_files:
        if file.filename.strip() == "":
            continue

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        file_paths.append(save_path)

    # Run your similarity processor
    report_html, report_csv = process_documents(file_paths)

    # Only send filename to HTML
    report_csv_filename = os.path.basename(report_csv)

    return render_template(
        "done.html",
        report_html=report_html,
        report_csv=report_csv_filename
    )

if __name__ == "__main__":
    app.run(debug=True)
