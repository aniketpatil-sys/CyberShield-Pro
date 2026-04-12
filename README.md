⚙️ Installation & Setup
Follow these steps to set up the project on your local machine:


1. Clone the Repository:

Bash

git clone [https://github.com/aniketpatil-sys/CyberShield-Pro.git](https://github.com/aniketpatil-sys/CyberShield-Pro.git)

cd CyberShield-Pro


2. Install Dependencies:

Bash
pip install -r requirements.txt


3. Configure Environment Variables:

Locate the .env.example file.

Rename it to .env

Enter your VirusTotal API Key.

Set your Admin Credentials (User & Password).


4. Run the Application:

Bash
python app.py
The application will start at: http://localhost:5000

open admin panel and go to model control and retrain your model
Admin panel will start at: http://localhost:5000/admin


📊 Technical Stack
Backend: Python, Flask

Machine Learning: Scikit-learn (Random Forest Classifier)

Database: SQLite3

Frontend: HTML5, CSS3, JavaScript

API Integration: VirusTotal API v3


📝 Disclaimer
This project is intended for educational and ethical security research purposes only. The developer is not responsible for any misuse of this tool.
