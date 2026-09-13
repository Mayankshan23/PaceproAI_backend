"""
PacePro AI - Comprehensive Backend Viva & Faculty Defense Master Guide Generator
================================================================================
Generates an authoritative, beautifully structured A4 PDF covering 0 to Advanced
questions (Ant to Dinosaur) that external examiners and faculty ask for backend developers.
"""

import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to add running headers and total page count footer."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Omit headers/footers on page 1 (Cover)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F172A"))
            self.drawString(40, 805, "PACEPRO AI — BACKEND & VIVA DEFENSE MASTER MANUAL")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(555, 805, "Level 0 → Level 5 (Ant to Dinosaur)")
            
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(40, 798, 555, 798)
            
            # Footer
            self.line(40, 45, 555, 45)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(40, 32, "Confidential • Final Year Computer Science / AI Defense Guide")
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(555, 32, page_str)
            
        self.restoreState()


def build_pdf(filename="PacePro_Backend_Viva_0_to_Advanced_Guide.pdf"):
    pdf_path = os.path.abspath(filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=55,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubTitle',
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#475569'),
        alignment=0
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0369A1'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )

    q_style = ParagraphStyle(
        'Question_Style',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    ans_style = ParagraphStyle(
        'Answer_Style',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'Code_Snippet',
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0F172A')
    )

    faculty_tip_style = ParagraphStyle(
        'Faculty_Tip',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#854D0E')
    )

    story = []

    # ==========================================
    # COVER / HEADER
    # ==========================================
    story.append(Paragraph("PACEPRO AI — BACKEND MASTER VIVA GUIDE", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("From Absolute Fundamentals (Ant / 0) to Advanced System Engineering (Dinosaur / 5)", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

    # Badge metadata table
    meta_data = [
        [
            Paragraph("<b>Target Project:</b> PacePro AI (FastAPI, OpenCV, MediaPipe, LLM)", body_style),
            Paragraph("<b>Focus:</b> Backend Developer & System Defense", body_style)
        ],
        [
            Paragraph("<b>Examiner Level:</b> College External / University Faculty", body_style),
            Paragraph("<b>Coverage:</b> Code syntax, Math, Networking, CV, DB, LLM", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Intro Note
    intro_p = Paragraph(
        "<b>How to use this manual:</b> When faculty take your viva, they test whether you actually wrote the backend code or just downloaded a repo. "
        "They intentionally start with deceptively simple atomic questions (<i>'What does this line mean?'</i>) and quickly escalate to complex systems questions (<i>'How does your vector math avoid parallax distortion in 3D?'</i>). "
        "This guide equips you with exact, crystal-clear answers across all 6 progressive difficulty tiers.",
        body_style
    )
    story.append(intro_p)
    story.append(Spacer(1, 10))

    # ==========================================
    # HELPER FUNCTION FOR Q&A CARDS
    # ==========================================
    def add_qa(q_num, level_tag, question, code_snippet, answer_text, faculty_tip):
        header_text = f"<b>Q{q_num} [{level_tag}]: {question}</b>"
        items = [Paragraph(header_text, q_style)]
        
        if code_snippet:
            code_p = Paragraph(code_snippet.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)
            code_table = Table([[code_p]], colWidths=[490])
            code_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            items.append(Spacer(1, 2))
            items.append(code_table)
            items.append(Spacer(1, 3))

        items.append(Paragraph(f"<b>Answer:</b> {answer_text}", ans_style))
        
        if faculty_tip:
            tip_p = Paragraph(f"<b>Faculty Killer Tip:</b> {faculty_tip}", faculty_tip_style)
            tip_table = Table([[tip_p]], colWidths=[490])
            tip_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF9C3')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#EAB308')),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            items.append(tip_table)
            
        items.append(Spacer(1, 8))
        story.append(KeepTogether(items))

    # ==========================================
    # LEVEL 0: THE ANT (Atomic Basics & Python Syntax)
    # ==========================================
    story.append(Paragraph("LEVEL 0: THE 'ANT' — ATOMIC PYTHON & PROGRAMMING FUNDAMENTALS", h1_style))
    story.append(Paragraph("<i>Faculty ask these to check if you have core computer science literacy or are memorizing.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        1, "Level 0",
        "If you write `a = 0` in Python, what actually happens in memory? Explain every token.",
        "a = 0",
        "<b>`a`</b> is an identifier (variable reference name). <b>`=`</b> is the assignment operator. <b>`0`</b> is an integer literal object. "
        "In Python, variables do not store values directly; they store references (memory addresses/pointers) to objects. Python allocates an Integer object `0` in memory (or retrieves it from the small integer cache pool [-5 to 256]) and binds the reference pointer `a` to that memory address.",
        "Faculty love asking 'Is Python pass-by-value or pass-by-reference?'. Answer: <i>'Python is Pass-by-Object-Reference (or Pass-by-Assignment) — immutable objects like int/str act like pass-by-value, mutable objects like lists/dicts act like pass-by-reference.'</i>"
    )

    add_qa(
        2, "Level 0",
        "What is the difference between a Python List, Tuple, and Dictionary? Where are they used in your backend?",
        "landmarks_list = [landmark1, landmark2] # Mutable\nANGLE_RANGES = (150.0, 165.0)           # Immutable Tuple\nplayer_metrics = {'knee_angle': 162.4}   # Key-Value Dict",
        "1. <b>List:</b> Ordered, mutable, dynamic array (e.g. collecting video frame angles over time).<br/>"
        "2. <b>Tuple:</b> Ordered, immutable, fixed size (e.g. constant angle threshold pairs like `(min, max)` to prevent accidental mutation).<br/>"
        "3. <b>Dictionary:</b> Hash-table based Key-Value store with O(1) average lookup (e.g. returning JSON responses and metrics mapping).",
        "Mention that Tuples are hashable and memory-efficient compared to lists because they don't over-allocate memory for dynamic resizing."
    )

    add_qa(
        3, "Level 0",
        "What are `*args` and `**kwargs` in Python functions?",
        "def compute_metrics(video_id, *flags, **extra_params):\n    # *flags -> tuple of positional args\n    # **extra_params -> dict of keyword args",
        "<b>`*args`</b> allows a function to accept any number of positional arguments as a <b>Tuple</b>. "
        "<b>`**kwargs`</b> allows accepting arbitrary named/keyword arguments as a <b>Dictionary</b>. "
        "In our backend, we use `**kwargs` in database managers and LLM wrappers to pass dynamic optional parameters without breaking existing function signatures.",
        "State: <i>'The single asterisk unpacks iterables, the double asterisk unpacks dictionaries.'</i>"
    )

    add_qa(
        4, "Level 0",
        "What is the difference between `try / except / finally`? When does `finally` execute?",
        "try:\n    cap = cv2.VideoCapture(video_path)\n    # process frames\nexcept Exception as e:\n    logger.error(f'Error: {e}')\nfinally:\n    cap.release() # ALWAYS executes",
        "<b>`try`</b> encapsulates risky operations that might throw exceptions. <b>`except`</b> catches and handles specific runtime errors. "
        "<b>`finally`</b> is a cleanup block that is <b>guaranteed to execute</b> whether an exception occurred, was handled, or even if the function hit an early `return` statement. In our video pipeline, `finally` guarantees camera/video memory release.",
        "Faculty will ask: <i>'What if there is a return in the try block?'</i> Answer: <i>'The finally block will STILL execute right before the return value is actually returned to the caller.'</i>"
    )

    # ==========================================
    # LEVEL 1: WEB FRAMEWORK & FASTAPI NETWORKING
    # ==========================================
    story.append(Paragraph("LEVEL 1: WEB API & FASTAPI BACKEND ARCHITECTURE", h1_style))
    story.append(Paragraph("<i>Faculty ask these to verify your API design, HTTP understanding, and asynchronous handling.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        5, "Level 1",
        "Why did you choose FastAPI over Flask or Django for PacePro AI?",
        "from fastapi import FastAPI\napp = FastAPI(title='PacePro AI Engine')",
        "1. <b>Native Asynchronous (ASGI):</b> FastAPI runs on Starlette and Uvicorn with native `async/await` event loops, handling thousands of concurrent requests without thread-blocking.<br/>"
        "2. <b>High Performance:</b> Built on Starlette and Pydantic, matching NodeJS/Go speed benchmarks.<br/>"
        "3. <b>Automatic Type Validation & Serialization:</b> Pydantic automatically validates request bodies and returns structured HTTP 422 errors on invalid types.<br/>"
        "4. <b>Auto-generated Swagger/OpenAPI Docs:</b> `/docs` interactive UI comes out of the box.",
        "Flask is synchronous (WSGI) by default and requires external plugins for data validation and Swagger. Django is too heavy for microservice CV/AI APIs."
    )

    add_qa(
        6, "Level 1",
        "What is CORS, why does CORS error happen, and how did you configure it?",
        "app.add_middleware(\n    CORSMiddleware,\n    allow_origins=['*'],\n    allow_credentials=True,\n    allow_methods=['*'],\n    allow_headers=['*'],\n)",
        "<b>CORS (Cross-Origin Resource Sharing)</b> is a browser security mechanism enforcing the Same-Origin Policy (SOP). "
        "When React on `http://localhost:5173` calls FastAPI on `http://localhost:8000`, the browser blocks it because the <b>port differs (different origin)</b>. "
        "The browser first sends an HTTP `OPTIONS` preflight request. We configured FastAPI `CORSMiddleware` to return `Access-Control-Allow-Origin: *` headers to allow cross-origin requests.",
        "Explain that CORS is a <i>browser-only</i> security check. Postman or Python scripts making requests to your backend never trigger CORS."
    )

    add_qa(
        7, "Level 1",
        "What is the difference between `async def` and normal `def` in FastAPI endpoints?",
        "@app.get('/health')\nasync def health_check():\n    return {'status': 'ok'}\n\n@app.post('/api/analyze-bowling')\ndef process_heavy_cv(file: UploadFile):\n    # CPU-bound OpenCV workload",
        "<b>`async def`</b> endpoints run directly on the main <b>asyncio Event Loop</b>. They are optimal for I/O-bound tasks (database queries, network calls to LLMs) where `await` yields CPU control to other requests.<br/>"
        "<b>Normal `def`</b> endpoints are automatically offloaded by FastAPI to an internal <b>thread pool</b> (`anyio.to_thread`). This prevents heavy CPU-blocking calculations (like OpenCV frame processing) from freezing the entire server event loop.",
        "Emphasize: <i>'If you put a synchronous CPU-heavy loop inside `async def` without thread-offloading, you block the entire single-threaded event loop for all users!'</i>"
    )

    add_qa(
        8, "Level 1",
        "What is Pydantic and how does `BaseModel` help in your backend?",
        "class PlanGenerationRequest(BaseModel):\n    user_id: Optional[str] = None\n    age: int = Field(..., ge=12, le=60)\n    weight_kg: float\n    bowling_pace_category: str",
        "<b>Pydantic</b> is a Python data validation and parsing library using standard Python type hints. "
        "When a client sends a JSON body, Pydantic parses strings to integers/floats, validates constraints (e.g. `ge=12` means greater than or equal to 12), strips malicious fields, and converts data into strongly typed Python objects with auto-generated schemas.",
        "If a user sends `age: 'abc'`, Pydantic intercepts it and immediately returns a clean `422 Unprocessable Entity` JSON error with the exact field location."
    )

    # ==========================================
    # LEVEL 2: COMPUTER VISION & PIPELINE PROCESSING
    # ==========================================
    story.append(Paragraph("LEVEL 2: COMPUTER VISION & MEDIAPIPE BIOMECHANICS", h1_style))
    story.append(Paragraph("<i>Faculty ask these to test how your system extracts poses, frames, and video streams.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        9, "Level 2",
        "How does a video stream get processed from React upload to OpenCV in Python?",
        "# FastAPI receives multipart/form-data\nwith tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:\n    shutil.copyfileobj(uploaded_file.file, tmp)\n    temp_path = tmp.name\n\ncap = cv2.VideoCapture(temp_path)\nfps = cap.get(cv2.CAP_PROP_FPS)\nwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))",
        "1. React sends `FormData` containing the binary video files via HTTP POST `multipart/form-data`.<br/>"
        "2. FastAPI receives it as an `UploadFile` spooled stream.<br/>"
        "3. We stream the binary buffer into a secure temp file (`tempfile.NamedTemporaryFile`).<br/>"
        "4. OpenCV `cv2.VideoCapture` opens the file descriptor, extracts video metadata (FPS, frame count, width, height), and iterates through frames in a decoding loop.",
        "Explain why we write to a temporary file: <i>'OpenCV's C++ backend requires a seekable file path or raw memory pointer; streaming raw HTTP chunks directly into `cv2.VideoCapture` is not natively supported.'</i>"
    )

    add_qa(
        10, "Level 2",
        "What is MediaPipe Pose Landmarker? How many landmarks does it detect and what coordinates does it provide?",
        "# MediaPipe outputs 33 3D landmarks per body\nlandmark = results.pose_landmarks[0][16] # Wrist\nx = landmark.x  # Normalized [0.0, 1.0] by image width\ny = landmark.y  # Normalized [0.0, 1.0] by image height\nz = landmark.z  # Relative landmark depth\nvisibility = landmark.visibility # Confidence score [0.0, 1.0]",
        "<b>MediaPipe Pose</b> is a two-stage deep learning pipeline (BlazePose detector + Landmark model) that detects <b>33 3D skeletal keypoints</b> across the human body in real-time. "
        "Coordinates $(x, y)$ are normalized between $0.0$ and $1.0$. $z$ represents relative depth where the origin $(0,0,0)$ is centered at the hip midpoint. Each keypoint includes `visibility` and `presence` confidence scores.",
        "Mention landmarks relevant to cricket bowling: #11 & #12 (Shoulders), #13 & #14 (Elbows), #15 & #16 (Wrists), #23 & #24 (Hips), #25 & #26 (Knees), #27 & #28 (Ankles)."
    )

    add_qa(
        11, "Level 2",
        "Why does OpenCV use BGR color order while MediaPipe and Web require RGB?",
        "# BGR to RGB conversion for MediaPipe\nrgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)",
        "OpenCV was created in 1999 when BGR (Blue-Green-Red) was the standard hardware format for camera sensors and Windows bitmap graphics. "
        "Modern deep learning frameworks and web standards use RGB. If we pass raw OpenCV frames directly to MediaPipe without `cv2.cvtColor`, blue and red channels are inverted, causing poor neural network keypoint confidence.",
        "Say: <i>'We convert BGR -> RGB before inference, and RGB -> BGR before saving annotated frames with OpenCV's VideoWriter.'</i>"
    )

    # ==========================================
    # LEVEL 3: BIOMECHANICAL MATHEMATICS & ALGORITHMS
    # ==========================================
    story.append(Paragraph("LEVEL 3: BIOMECHANICAL VECTOR MATHEMATICS & PHASE DETECTION", h1_style))
    story.append(Paragraph("<i>Faculty ask these to verify your pure mathematical derivations and angle calculation logic.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        12, "Level 3",
        "What is the mathematical formula used to calculate the 3D joint angle between 3 body landmarks (e.g. Knee Angle)?",
        "import numpy as np\n\ndef calculate_angle(p_hip, p_knee, p_ankle):\n    v1 = np.array(p_hip) - np.array(p_knee)    # Vector Knee -> Hip\n    v2 = np.array(p_ankle) - np.array(p_knee)  # Vector Knee -> Ankle\n    \n    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))\n    cosine = np.clip(cosine, -1.0, 1.0) # Prevent numerical float precision error\n    return np.degrees(np.arccos(cosine))",
        "Let vertex joint be $B$ (Knee) between points $A$ (Hip) and $C$ (Ankle):<br/>"
        "1. Create vectors: $\\vec{u} = A - B$ and $\\vec{v} = C - B$<br/>"
        "2. Compute Dot Product: $\\vec{u} \\cdot \\vec{v} = u_x v_x + u_y v_y + u_z v_z$<br/>"
        "3. Compute Magnitudes (Norms): $\\|\\vec{u}\\| = \\sqrt{u_x^2 + u_y^2 + u_z^2}$ and $\\|\\vec{v}\\| = \\sqrt{v_x^2 + v_y^2 + v_z^2}$<br/>"
        "4. Angle in radians: $\\theta = \\arccos\\left(\\frac{\\vec{u} \\cdot \\vec{v}}{\\|\\vec{u}\\| \\|\\vec{v}\\|}\\right)$ and convert to degrees $(\\theta \\times \\frac{180}{\\pi})$.",
        "Faculty will ask: <i>'Why do you do np.clip(cosine, -1.0, 1.0)?'</i> Answer: <i>'Floating-point rounding errors can result in 1.0000000002, which causes arccos() to crash with NaN/math domain error!'</i>"
    )

    add_qa(
        13, "Level 3",
        "How does the algorithm automatically detect the 4 key bowling phases: BFC, FFC, Release, and Follow-through?",
        "# Phase Detection Logic in Pipeline:\n# 1. Back Foot Contact (BFC): Local peak in back ankle deceleration + lowest hip height\n# 2. Front Foot Contact (FFC): Max distance between ankles + lead knee impact spike\n# 3. Ball Release (BR): Max bowling wrist height/velocity relative to shoulder\n# 4. Follow-Through: Deceleration and torso forward flexion",
        "1. <b>BFC (Back Foot Contact):</b> Detected when the trailing foot strikes the ground — identified by monitoring the vertical trajectory ($y$-coordinate minima) and velocity zero-crossings of the back ankle landmark.<br/>"
        "2. <b>FFC (Front Foot Contact):</b> Detected by finding the maximum stride length (Euclidean distance between left and right ankles) combined with front foot impact deceleration.<br/>"
        "3. <b>BR (Ball Release):</b> The frame where the bowling wrist reaches peak vertical height and angular velocity above the shoulder girdle.<br/>"
        "4. <b>Follow-Through:</b> The rotational and forward flexion deceleration phase after ball release.",
        "This is done using time-series peak-detection algorithms (`scipy.signal.find_peaks` or moving window derivatives) over normalized landmark coordinates across all frames."
    )

    add_qa(
        14, "Level 3",
        "What is ICC legal bowling arm extension limit and how does PacePro evaluate elbow hyperextension?",
        "# ICC Law 21.2: Maximum allowable elbow flexion/extension during delivery is 15 degrees\nelbow_angle_at_release = calculate_angle(shoulder, elbow, wrist)",
        "Under <b>ICC Law 21.2</b> (Fair Delivery - The Arm), a bowler is permitted a maximum of <b>15 degrees of elbow extension</b> from the horizontal shoulder position until ball release. "
        "PacePro tracks the bowling elbow angle continuously from the moment the bowling arm reaches horizontal until release. If $\\Delta \\theta_{\\text{elbow}} = |\\theta_{\\text{release}} - \\theta_{\\text{horizontal}}| > 15^\\circ$, it flags potential illegal throwing/chucking risk.",
        "Clarify that static single angles aren't illegal; it is the <i>dynamic straightening range</i> during the delivery swing that is regulated."
    )

    # ==========================================
    # LEVEL 4: ATHLETIC ACWR & NUTRITION ENGINE
    # ==========================================
    story.append(Paragraph("LEVEL 4: WORKLOAD (ACWR) & NUTRITION ALGORITHMS", h1_style))
    story.append(Paragraph("<i>Faculty ask these to test your domain-specific sports science calculations and nutrition logic.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        15, "Level 4",
        "What is ACWR (Acute:Chronic Workload Ratio) and what is the exact formula implemented?",
        "# ACWR Calculation Implementation:\nacute_workload = sum(overs_last_7_days)           # 7-day fatigue load\nchronic_workload = sum(overs_last_28_days) / 4.0   # 28-day fitness baseline\n\nacwr = acute_workload / max(chronic_workload, 1.0) # Avoid division by zero",
        "<b>ACWR</b> was developed by Dr. Tim Gabbett to predict injury risk in fast bowlers by balancing fatigue vs fitness baseline:<br/>"
        "• <b>Acute Load (7 Days):</b> Represents short-term fatigue accumulation.<br/>"
        "• <b>Chronic Load (28 Days):</b> Average weekly load over 4 weeks representing chronic physical conditioning.<br/>"
        "• <b>Interpretation:</b><br/>"
        "&nbsp;&nbsp;- <b>0.8 to 1.3:</b> <i>'Sweet Spot'</i> (High fitness, lowest relative injury risk).<br/>"
        "&nbsp;&nbsp;- <b>&gt; 1.5:</b> <i>'Danger Zone'</i> (Workload spiked too fast; injury risk spikes by 2-4x — lumbar stress fracture risk).<br/>"
        "&nbsp;&nbsp;- <b>&lt; 0.8:</b> <i>'Under-training'</i> (Athlete unconditioned, prone to injury when intensity rises).",
        "Mention that Fast Bowlers are especially susceptible to L4/L5 lumbar spondylolysis when ACWR > 1.5."
    )

    add_qa(
        16, "Level 4",
        "How does the backend compute Calorie Targets (TDEE) and Macronutrient Breakdown for Fast Bowlers?",
        "# Mifflin-St Jeor Formula for BMR (Male):\n# BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5\n# TDEE = BMR * Physical_Activity_Level (e.g. 1.725 for fast bowlers)\n# Protein: 1.8 to 2.2 g/kg bodyweight for myofibrillar repair\n# Fats: 25% of total caloric intake\n# Carbs: Remaining calories to replenish glycogen stores",
        "1. <b>BMR (Basal Metabolic Rate):</b> Computed via Mifflin-St Jeor or Harris-Benedict equation.<br/>"
        "2. <b>TDEE (Total Daily Energy Expenditure):</b> $\\text{TDEE} = \\text{BMR} \\times \\text{PAL factor}$ (1.725 for heavy bowling & sprint training).<br/>"
        "3. <b>Protein:</b> Set at $1.8 - 2.2\\text{ g/kg}$ of body weight (4 kcal/g) for eccentric muscle damage recovery.<br/>"
        "4. <b>Fats:</b> Set at $20-25\\%$ of total calories (9 kcal/g) for steroid hormone/testosterone synthesis.<br/>"
        "5. <b>Carbohydrates:</b> Remaining caloric intake (4 kcal/g) allocated to carbs for fast glycolytic energy replenishment.",
        "Faculty tip: <i>'Fast bowlers require high protein due to the massive ground reaction forces (up to 8-9x body weight at Front Foot Contact) causing high muscle micro-trauma.'</i>"
    )

    # ==========================================
    # LEVEL 5: THE DINOSAUR (Advanced Engineering & AI)
    # ==========================================
    story.append(Paragraph("LEVEL 5: THE 'DINOSAUR' — ENTERPRISE CONCURRENCY, LLM & SYSTEM SCALING", h1_style))
    story.append(Paragraph("<i>Faculty ask these high-level architectural questions for highest grade marks and honors evaluation.</i>", body_style))
    story.append(Spacer(1, 4))

    add_qa(
        17, "Level 5",
        "What is the Python GIL (Global Interpreter Lock)? How does it affect your video processing and server throughput?",
        "# Python GIL prevents multiple native threads from executing Python bytecodes simultaneously in CPython.",
        "The <b>GIL</b> is a mutex that protects access to Python objects, preventing race conditions by allowing only one native thread to execute Python bytecode at a time. "
        "<b>Impact on PacePro:</b><br/>"
        "1. <b>I/O Bound (FastAPI network/database/LLM):</b> GIL is released during I/O wait, so `asyncio` or multithreading gives true high-throughput concurrency.<br/>"
        "2. <b>CPU Bound (OpenCV image matrix calculations & NumPy):</b> Heavy NumPy and OpenCV C++ routines <i>release the GIL</i> internally during matrix arithmetic, allowing multi-core CPU acceleration.<br/>"
        "3. For true parallel multi-user video decoding, we scale with <b>multi-worker processes</b> (e.g. Uvicorn with `--workers 4`).",
        "State: <i>'AsyncIO solves I/O concurrency; Multi-processing solves CPU parallelism.'</i>"
    )

    add_qa(
        18, "Level 5",
        "How is your LLM AI Coach integrated? What happens if the API key fails or rate limits are reached?",
        "# Resilient Fallback Chain in llm_analyzer.py & config.py:\n# 1. Primary: Groq / Gemini (Fast & High Token Limit)\n# 2. Secondary: OpenRouter / Mistral API\n# 3. Ultimate Fallback: Deterministic Rule-Based Biomechanical Expert System",
        "We implement a <b>Multi-Tier Resilient Fallback Architecture</b>:<br/>"
        "• <b>Primary:</b> High-speed LLM (Groq LLaMA-3.3-70B or Google Gemini 2.5 Flash) with strict temperature ($0.2-0.3$) and structured JSON response schemas.<br/>"
        "• <b>Secondary Fallback:</b> If HTTP 429 (Rate Limit) or 500 occurs, the `llm_analyzer` automatically falls back to an alternative configured API key.<br/>"
        "• <b>Zero-Crash Safety Net:</b> If all internet APIs are down, an internal <b>Heuristic Rule-Based Expert Engine</b> synthesizes human-quality coaching feedback based on calculated angle standard deviations without failing the user's video analysis request.",
        "Mention that prompts use <i>Few-Shot Grounding</i> and pass quantified numerical degrees rather than raw video bytes to save tokens and avoid LLM hallucinations."
    )

    add_qa(
        19, "Level 5",
        "How does Multi-View Fusion (Front View + Side View 1 + Side View 2) work in PacePro?",
        "# Multi-view fusion synchronizes distinct camera angles:\n# 1. Temporal Sync: Aligning BFC and Release frame timestamps\n# 2. Spatial Fusion: Front View measures Lateral Trunk Flexion; Side View measures Knee Flexion & Arm Extension\n# 3. 3D Pose Reconstruction: Combines 2D views to eliminate single-camera perspective parallax.",
        "Single-camera analysis suffers from <b>perspective foreshortening</b> (e.g., viewing an elbow from the front makes a bent arm look straight).<br/>"
        "• <b>Side View (Sagittal Plane):</b> Accurately measures Knee Flexion, Hip Hinge, and Elbow Angle.<br/>"
        "• <b>Front View (Coronal Plane):</b> Accurately measures Lateral Trunk Flexion, Bowling Arm Release Height, and Stride Alignment.<br/>"
        "• <b>Fusion Engine:</b> Temporally synchronizes the video streams based on the global impact timestamp (Front Foot Contact) and merges the angle vectors into a comprehensive 360-degree Biomechanical Scorecard.",
        "Emphasize: <i>'Multi-view fusion transforms 2D camera limitations into a high-fidelity 3D kinematic assessment.'</i>"
    )

    add_qa(
        20, "Level 5",
        "How would you scale this backend to 100,000 active bowlers in production?",
        "# Production Architecture Roadmap:\n# 1. Frontend: Cloudflare CDN / Vercel Edge\n# 2. API Gateway: NGINX Reverse Proxy + SSL Termination\n# 3. Async Task Queue: Celery + Redis for Video Processing Worker Nodes\n# 4. Storage: AWS S3 with Pre-signed URLs for direct video upload\n# 5. Database: MongoDB Atlas / PostgreSQL with read-replicas and connection pooling\n# 6. Model Acceleration: ONNX Runtime / TensorRT with NVIDIA GPU acceleration (T4/A10G)",
        "1. <b>Decouple Video Processing:</b> Instead of processing videos inside the HTTP request thread, the API immediately enqueues a job into <b>Celery / RabbitMQ / Redis</b> and returns `202 Accepted` with a `job_id`.<br/>"
        "2. <b>GPU Worker Pool:</b> Background workers running ONNX Runtime on auto-scaling GPU spot instances consume the queue.<br/>"
        "3. <b>Direct-to-S3 Uploads:</b> React uploads large MP4 files directly to AWS S3 via pre-signed URLs, preventing video traffic from clogging API servers.<br/>"
        "4. <b>WebSockets / Server-Sent Events (SSE):</b> Push real-time progress updates (20%, 60%, 100%) to the user's browser.",
        "This is the golden answer that proves to examiners that you possess enterprise-grade system design knowledge."
    )

    # ==========================================
    # FINAL CHEAT SHEET / FORMULA TABLE
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("QUICK REFERENCE: BIOMECHANICAL BENCHMARKS CHEAT SHEET", h1_style))
    story.append(Paragraph("Memorize these exact angles before entering your viva room:", body_style))
    story.append(Spacer(1, 6))

    cheat_data = [
        [Paragraph("<b>Metric / Parameter</b>", q_style), Paragraph("<b>Phase</b>", q_style), Paragraph("<b>Optimal Range</b>", q_style), Paragraph("<b>Clinical / Performance Risk</b>", q_style)],
        [Paragraph("Back Knee Flexion", body_style), Paragraph("BFC", body_style), Paragraph("150° – 165°", body_style), Paragraph("Collapsing back knee loses momentum", body_style)],
        [Paragraph("Front Knee Angle", body_style), Paragraph("FFC", body_style), Paragraph("155° – 175° (Braced)", body_style), Paragraph("Bent knee (<140°) dissipates ground reaction force", body_style)],
        [Paragraph("Lateral Trunk Flexion", body_style), Paragraph("BR (Release)", body_style), Paragraph("10° – 25°", body_style), Paragraph(">35° causes severe Lumbar Stress Fractures", body_style)],
        [Paragraph("Elbow Extension Range", body_style), Paragraph("Delivery Swing", body_style), Paragraph("< 15° (ICC Rule)", body_style), Paragraph(">15° is classified as illegal bowling (Chucking)", body_style)],
        [Paragraph("Shoulder-Hip Separation", body_style), Paragraph("BFC to FFC", body_style), Paragraph("20° – 35°", body_style), Paragraph("Excessive counter-rotation causes disc herniation", body_style)],
        [Paragraph("ACWR Ratio", body_style), Paragraph("Weekly Monitoring", body_style), Paragraph("0.8 – 1.3", body_style), Paragraph(">1.5 causes acute overtraining & soft-tissue injury", body_style)]
    ]

    cheat_table = Table(cheat_data, colWidths=[130, 60, 110, 190])
    cheat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284C7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(cheat_table)

    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>Examiner Golden Rule:</b> Always answer with confidence, define terms before diving into code, and connect the computer science technique (e.g. dot products, async I/O, Pydantic) back to the real-world domain problem (fast bowler injury prevention & speed optimization). Good luck with your defense!", body_style))

    # Build the document with custom numbered canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Successfully generated at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    out_pdf = build_pdf()
    print("Generation finished.")
