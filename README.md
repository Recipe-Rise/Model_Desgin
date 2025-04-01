# 🍳 Recipe Personalization Model

A smart recipe recommendation system that personalizes recipe suggestions based on user preferences, dietary restrictions, and fitness goals.

## 🌟 Features

- **Personalized Recommendations**: Get recipe suggestions based on your:
  - Dietary preferences and restrictions
  - Fitness goals (weight loss, muscle gain, maintenance)
  - Cooking time preferences
  - Calorie targets
  - Meal history and ratings

- **Nutritional Tracking**:
  - Detailed macronutrient breakdown
  - Calorie tracking
  - Sugar content monitoring (especially for diabetic users)
  - Portion size recommendations

- **User-Friendly Interface**:
  - Interactive Streamlit dashboard
  - Beautiful recipe cards
  - Step-by-step cooking instructions
  - Ingredient lists
  - Nutritional visualizations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Recipe-Rise/Model_Desgin.git
cd Model_Desgin
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 🛠️ Project Structure

```
Model_Desgin/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions and helper methods
├── style.css           # Custom styling for the web interface
├── requirements.txt    # Python dependencies
├── data/              # Recipe dataset
├── images/            # Application images and assets
└── notebooks/         # Jupyter notebooks for analysis
```

## 📊 Data Analysis

The project includes comprehensive Exploratory Data Analysis (EDA) notebooks that cover:
- Recipe feature analysis
- Nutritional content analysis
- Ingredient pattern analysis
- Cooking time and complexity metrics

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

---

Made with ❤️ for better cooking experiences
