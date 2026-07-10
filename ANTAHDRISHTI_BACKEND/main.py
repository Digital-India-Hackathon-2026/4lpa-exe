from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from datetime import datetime
from pathlib import Path
import shutil
import uuid
import os
import cv2


# ---------------------------------------------------------
# APPLICATION SETUP
# ---------------------------------------------------------

app = FastAPI(
    title="ANTAHDRISHTI",
    description="AI-powered infrastructure defect detection platform",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"
REPORT_FOLDER = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER.mkdir(exist_ok=True)


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"best.pt was not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# Stores the latest result for PDF report generation
latest_analysis = {
    "filename": "No inspection completed",
    "detections": [],
    "priority": "UNKNOWN",
    "result_image": None
}


# ---------------------------------------------------------
# HOME ROUTE
# ---------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(str(FRONTEND_FILE))



# ---------------------------------------------------------
# ANALYZE IMAGE USING BEST.PT
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    global latest_analysis

    # Create a safe unique filename
    original_extension = Path(file.filename).suffix.lower()

    if original_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        return {
            "status": "error",
            "message": "Please upload a JPG, JPEG, PNG or WEBP image."
        }

    unique_name = f"{uuid.uuid4().hex}{original_extension}"
    upload_path = UPLOAD_FOLDER / unique_name

    # Save uploaded image
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=str(upload_path),
        conf=0.25,
        save=False,
        verbose=False
    )

    detections = []
    total_confidence = 0.0

    result = results[0]

    # Read bounding-box detections
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names[class_id]

            coordinates = box.xyxy[0].tolist()

            detection = {
                "defect": class_name,
                "confidence": round(confidence * 100, 2),
                "bounding_box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2)
                }
            }

            detections.append(detection)
            total_confidence += confidence

    # Draw YOLO detections on image
    plotted_image = result.plot()

    result_filename = f"detected_{unique_name}"
    result_path = RESULT_FOLDER / result_filename

    cv2.imwrite(str(result_path), plotted_image)

    # Simple preliminary maintenance priority
    defect_count = len(detections)

    if defect_count == 0:
        priority = "LOW"
        recommendation = "No visible defect detected. Continue routine monitoring."

    elif defect_count == 1:
        priority = "MEDIUM"
        recommendation = "Engineer review recommended during the next inspection."

    elif defect_count <= 3:
        priority = "HIGH"
        recommendation = "Schedule a detailed engineering inspection."

    else:
        priority = "CRITICAL"
        recommendation = "Immediate engineer inspection and maintenance review required."

    average_confidence = (
        round((total_confidence / defect_count) * 100, 2)
        if defect_count > 0
        else 0
    )

    latest_analysis = {
        "filename": file.filename,
        "detections": detections,
        "priority": priority,
        "recommendation": recommendation,
        "average_confidence": average_confidence,
        "result_image": result_filename,
        "inspection_time": datetime.now().strftime("%d-%m-%Y %H:%M")
    }

    return {
        "status": "success",
        "original_filename": file.filename,
        "model_classes": model.names,
        "defect_count": defect_count,
        "detections": detections,
        "average_confidence": average_confidence,
        "preliminary_priority": priority,
        "recommendation": recommendation,
        "result_image_url": f"/result-image/{result_filename}",
        "note": "This is AI-assisted visual prioritization, not structural safety certification."
    }


# ---------------------------------------------------------
# VIEW DETECTED IMAGE
# ---------------------------------------------------------

@app.get("/result-image/{filename}")
def get_result_image(filename: str):

    file_path = RESULT_FOLDER / filename

    if not file_path.exists():
        return {
            "status": "error",
            "message": "Result image not found."
        }

    return FileResponse(str(file_path))


# ---------------------------------------------------------
# GENERATE PDF REPORT
# ---------------------------------------------------------

@app.get("/generate-report")
def generate_report():

    report_id = "INS-" + str(uuid.uuid4())[:8].upper()
    filename = f"{report_id}.pdf"
    filepath = REPORT_FOLDER / filename

    c = canvas.Canvas(str(filepath), pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 60, "ANTAHDRISHTI")

    c.setFont("Helvetica", 12)
    c.drawString(
        50,
        height - 85,
        "AI-Assisted Infrastructure Inspection Report"
    )

    c.line(50, height - 100, width - 50, height - 100)

    # Inspection details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 135, "Inspection Details")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 160, f"Inspection ID: {report_id}")
    c.drawString(
        50,
        height - 180,
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )
    c.drawString(
        50,
        height - 200,
        f"Image: {latest_analysis.get('filename', 'Unknown')}"
    )

    # Detected defects
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 240, "Detected Defects")

    detections = latest_analysis.get("detections", [])

    y = height - 270

    if not detections:
        c.setFont("Helvetica", 11)
        c.drawString(70, y, "No visible defects detected by the AI model.")
        y -= 25

    else:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y, "Defect")
        c.drawString(250, y, "Confidence")
        y -= 20

        for detection in detections:

            if y < 180:
                c.showPage()
                y = height - 60

            c.setFont("Helvetica", 10)
            c.drawString(70, y, str(detection["defect"]))
            c.drawString(
                250,
                y,
                f'{detection["confidence"]}%'
            )
            y -= 20

    # Priority
    y -= 20

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Preliminary Maintenance Priority")

    y -= 40

    priority = latest_analysis.get("priority", "UNKNOWN")

    if priority == "LOW":
        c.setFillColor(colors.green)
    elif priority == "MEDIUM":
        c.setFillColor(colors.orange)
    elif priority == "HIGH":
        c.setFillColor(colors.red)
    elif priority == "CRITICAL":
        c.setFillColor(colors.darkred)
    else:
        c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, y, priority)

    c.setFillColor(colors.black)

    # Recommendation
    y -= 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Recommendation")

    y -= 25

    c.setFont("Helvetica", 11)

    recommendation = latest_analysis.get(
        "recommendation",
        "Complete an AI inspection before generating the report."
    )

    c.drawString(70, y, recommendation)

    # Disclaimer
    c.setFont("Helvetica", 8)
    c.drawString(
        50,
        55,
        "This report provides AI-assisted visual prioritization only."
    )
    c.drawString(
        50,
        42,
        "Final structural assessment must be performed by a qualified engineer."
    )

    c.save()

    return FileResponse(
        str(filepath),
        media_type="application/pdf",
        filename=filename
    )