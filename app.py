import streamlit as st
from glob import glob
import pandas as pd
import utils
import plotly.graph_objects as go
from datetime import datetime

## STREAMLIT CONFIGURATION
## --------------------------------------------------------------------------------##
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍳",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

## Session state variables
## -------------------------------------------------------------------

if "data" not in st.session_state:
    files = glob(r"data/*.parquet")
    df = pd.read_parquet(files)
    df["ingredients"] = (
        df["ingredients"]
        .str.strip("[]")
        .str.replace("'", "")
        .str.replace('"', "")
        .str.split(",")
        .apply(lambda x: [y.strip() for y in x])
    )
    st.session_state["data"] = df

if "meal_history" not in st.session_state:
    st.session_state["meal_history"] = {
        "recipes": [],  # List of recipe names
        "ratings": {},  # Dictionary of recipe ratings (1-5)
        "last_cooked": {},  # Dictionary of last cooked dates
        "frequency": {},  # Dictionary of cooking frequency
        "preferences": {  # User preferences based on history
            "favorite_cuisines": set(),
            "favorite_ingredients": set(),
            "preferred_cooking_time": 0,
            "preferred_meal_size": 0
        }
    }

if "result" not in st.session_state:
    st.session_state["result"] = None

if "use_history" not in st.session_state:
    st.session_state["use_history"] = False

if "user_info" not in st.session_state:
    st.session_state["user_info"] = None

if "has_diabetes" not in st.session_state:
    st.session_state["has_diabetes"] = False

## App Layout
## -------------------------------------------------------------------

st.image(r"images/logo-no-background.png", width=400)

# User Information Section
st.header("User Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    activity_level = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"],
        help="Sedentary: Little or no exercise\nLightly Active: Light exercise 1-3 days/week\nModerately Active: Moderate exercise 3-5 days/week\nVery Active: Hard exercise 6-7 days/week\nExtra Active: Very hard exercise & physical job or training twice/day"
    )
    
    # Add preparation time filter
    max_prep_time = st.number_input(
        "Maximum preparation time (minutes)",
        min_value=0,
        max_value=int(st.session_state["data"]["minutes"].max()),
        value=60,
        step=5,
        help="Maximum time you want to spend preparing the recipe"
    )
    
    # Combined diabetes control
    has_diabetes = st.checkbox(
        "I have diabetes (will filter high-sugar recipes)",
        key="has_diabetes",  # This will automatically update the session state
        help="Check this to filter out recipes with high sugar content and receive diabetic-friendly recommendations"
    )
    
    if has_diabetes:
        st.info("""
        Diabetic-friendly filter limits sugar content to:
        - Women: ≤ 25g (6 teaspoons) per day
        - Men: ≤ 36g (9 teaspoons) per day
        
        This helps maintain stable blood sugar levels and follows ADA guidelines.
        """)

# Convert activity level to multiplier
activity_multipliers = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
    "Extra Active": 1.9
}

# Fitness Goal Selection
fitness_goal = st.selectbox(
    "Your Fitness Goal",
    ["Loss weight and gain muscles", "Loss weight", "Gain weight", "Gain muscles", "Fitness (maintain)"],
    help="Select your primary fitness goal to get personalized recipe recommendations"
)

# First, move the calorie input fields outside the columns for better visibility
st.subheader("Calorie Range")
col_min, col_max = st.columns(2)
with col_min:
    min_calories = st.number_input(
        label="Min Calories",
        min_value=0,
        max_value=2000,
        value=0,
        step=50,
        help="Minimum calories per serving"
    )
with col_max:
    max_calories = st.number_input(
        label="Max Calories",
        min_value=min_calories,  # Ensure max is not less than min
        max_value=2000,
        value=2000,
        step=50,
        help="Maximum calories per serving"
    )

# Calculate user's nutritional needs
if st.button("Calculate My Nutritional Needs"):
    # Validate inputs
    if weight <= 0 or height <= 0 or age <= 0:
        st.error("Please enter valid values for weight, height, and age.")
    else:
        bmr = utils.calculate_bmr(weight, height, age, gender)
        tdee = utils.calculate_tdee(bmr, activity_multipliers[activity_level])
        protein_target, carbs_target, fats_target = utils.get_macro_targets(tdee, fitness_goal, weight)
        
        # Calculate adjusted calories based on goal
        if fitness_goal == "Loss weight":
            adjusted_calories = tdee - 500
            calorie_explanation = "500 calorie deficit for steady weight loss"
        elif fitness_goal == "Loss weight and gain muscles":
            adjusted_calories = tdee - 300
            calorie_explanation = "300 calorie deficit for muscle-preserving weight loss"
        elif fitness_goal == "Gain weight":
            adjusted_calories = tdee + 500
            calorie_explanation = "500 calorie surplus for weight gain"
        elif fitness_goal == "Gain muscles":
            adjusted_calories = tdee + 300
            calorie_explanation = "300 calorie surplus for muscle gain"
        else:  # Fitness (maintain)
            adjusted_calories = tdee
            calorie_explanation = "Maintenance calories"
        
        st.session_state["user_info"] = {
            "bmr": bmr,
            "tdee": tdee,
            "adjusted_calories": adjusted_calories,
            "protein_target": protein_target,
            "carbs_target": carbs_target,
            "fats_target": fats_target
        }
        
        # Display the results in a more informative way
        st.success("Your Nutritional Targets Calculated!")
        
        # Display BMR and TDEE
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Basal Metabolic Rate (BMR)", f"{bmr:.0f} kcal")
        with col2:
            st.metric("Total Daily Energy Expenditure (TDEE)", f"{tdee:.0f} kcal")
        
        # Display adjusted calories with explanation
        st.subheader("Goal-Adjusted Calories")
        st.info(f"""
        **Target Daily Calories: {adjusted_calories:.0f} kcal**
        _{calorie_explanation}_
        """)
        
        # Display macronutrient targets
        st.subheader("Macronutrient Targets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            protein_cals = protein_target * 4
            protein_pct = (protein_cals / adjusted_calories) * 100
            st.metric(
                "Protein", 
                f"{protein_target:.0f}g",
                f"{protein_cals:.0f} kcal ({protein_pct:.0f}%)"
            )
            
        with col2:
            carbs_cals = carbs_target * 4
            carbs_pct = (carbs_cals / adjusted_calories) * 100
            st.metric(
                "Carbohydrates", 
                f"{carbs_target:.0f}g",
                f"{carbs_cals:.0f} kcal ({carbs_pct:.0f}%)"
            )
            
        with col3:
            fats_cals = fats_target * 9
            fats_pct = (fats_cals / adjusted_calories) * 100
            st.metric(
                "Fats", 
                f"{fats_target:.0f}g",
                f"{fats_cals:.0f} kcal ({fats_pct:.0f}%)"
            )
        
        # Add daily meal distribution suggestion
        st.subheader("Suggested Meal Distribution")
        st.info(f"""
        Based on your daily target of {adjusted_calories:.0f} kcal:
        - Breakfast: {(adjusted_calories * 0.25):.0f} kcal (25%)
        - Lunch: {(adjusted_calories * 0.35):.0f} kcal (35%)
        - Dinner: {(adjusted_calories * 0.30):.0f} kcal (30%)
        - Snacks: {(adjusted_calories * 0.10):.0f} kcal (10%)
        """)

# Add a section to display meal history with more details
if st.session_state["meal_history"]["recipes"]:
    st.subheader("Your Meal History")
    
    # Create columns for the history table
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        st.write("**Recipe Name**")
    with col2:
        st.write("**Calories**")
    with col3:
        st.write("**Rating**")
    with col4:
        st.write("**Last Cooked**")
    with col5:
        st.write("**Actions**")
    
    # Display each meal in history with details and controls
    for i, meal_name in enumerate(st.session_state["meal_history"]["recipes"]):
        meal_data = st.session_state["data"][st.session_state["data"]["name"] == meal_name].iloc[0]
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        with col1:
            st.write(meal_name)
        with col2:
            st.write(f"{meal_data['calories']} kcal")
        with col3:
            # Rating selector
            current_rating = st.session_state["meal_history"]["ratings"].get(meal_name, 0)
            new_rating = st.selectbox(
                "Rating",
                options=[0, 1, 2, 3, 4, 5],
                index=current_rating,
                key=f"rating_{i}",
                label_visibility="collapsed"
            )
            if new_rating != current_rating:
                st.session_state["meal_history"]["ratings"][meal_name] = new_rating
                st.experimental_rerun()
        with col4:
            last_cooked = st.session_state["meal_history"]["last_cooked"].get(meal_name, "Never")
            st.write(last_cooked)
        with col5:
            if st.button("🗑️", key=f"remove_{i}"):
                # Remove recipe and its associated data
                st.session_state["meal_history"]["recipes"].remove(meal_name)
                if meal_name in st.session_state["meal_history"]["ratings"]:
                    del st.session_state["meal_history"]["ratings"][meal_name]
                if meal_name in st.session_state["meal_history"]["last_cooked"]:
                    del st.session_state["meal_history"]["last_cooked"][meal_name]
                if meal_name in st.session_state["meal_history"]["frequency"]:
                    del st.session_state["meal_history"]["frequency"][meal_name]
                st.experimental_rerun()
    
    # Add toggle for history-based recommendations with explanation
    st.markdown("---")
    st.session_state["use_history"] = st.checkbox(
        "Use my meal history for recommendations",
        value=st.session_state["use_history"],
        help="When enabled, recommendations will be based on your ratings, cooking frequency, and preferences"
    )
    
    if st.session_state["use_history"]:
        st.info("""
        **Enhanced History-Based Recommendations**
        - Recommendations consider your ratings (1-5 stars)
        - Frequent and recently cooked recipes influence suggestions
        - Your preferred ingredients and cuisines are prioritized
        - Results are filtered to match your current fitness goals and dietary preferences
        """)

if st.session_state["use_history"]:
    recipe = None
else:
    recipe = st.selectbox(
        label="Search the recipe:", 
        options=st.session_state["data"]["name"], 
        index=100
    )

col1, col2 = st.columns([1, 3])
with col1:
    num_recipes = st.number_input(
        label="Number of similar recipes", min_value=1, max_value=10, step=1
    )

## Find similarity
## -------------------------------------------------------------------

def assign_values():
    try:
        macro_targets = None
        min_cal = min_calories  # Use user input directly
        max_cal = max_calories  # Use user input directly
        
        if st.session_state["user_info"] and not (min_calories or max_calories):
            # Only use TDEE-based calculations if user hasn't specified a calorie range
            tdee = st.session_state["user_info"]["tdee"]
            macro_targets = (
                st.session_state["user_info"]["protein_target"],
                st.session_state["user_info"]["carbs_target"],
                st.session_state["user_info"]["fats_target"]
            )
            
            # Calculate suggested calorie ranges based on goal
            if fitness_goal == "Loss weight":
                min_cal = int(tdee * 0.20)
                max_cal = int(tdee * 0.35)
            elif fitness_goal == "Loss weight and gain muscles":
                min_cal = int(tdee * 0.25)
                max_cal = int(tdee * 0.40)
            elif fitness_goal == "Gain weight":
                min_cal = int(tdee * 0.30)
                max_cal = int(tdee * 0.45)
            elif fitness_goal == "Gain muscles":
                min_cal = int(tdee * 0.25)
                max_cal = int(tdee * 0.40)
            else:  # Fitness (maintain)
                min_cal = int(tdee * 0.25)
                max_cal = int(tdee * 0.35)
            
            st.info(f"""
            **Suggested Calorie Range Per Meal**
            - Minimum: {min_cal:.0f} kcal
            - Maximum: {max_cal:.0f} kcal
            
            Based on your daily needs ({tdee:.0f} kcal) and goal ({fitness_goal}).
            You can adjust this range using the inputs above.
            """)
        else:
            # User has specified a calorie range, use it
            st.info(f"""
            **Selected Calorie Range**
            - Minimum: {min_cal} kcal
            - Maximum: {max_cal} kcal
            """)

        # Debug information
        st.write("Using calorie range:", min_cal, "-", max_cal, "kcal")
        
        if st.session_state["use_history"]:
            st.write("Using history-based recommendations")
            st.session_state["result"] = utils.get_recommendations_from_history(
                st.session_state["data"],
                st.session_state["meal_history"]["recipes"],
                num_recipes,
                min_calories=min_cal,
                max_calories=max_cal,
                diabetic_friendly=st.session_state.get("has_diabetes", False),
                macro_targets=macro_targets,
                fitness_goal=fitness_goal,
                max_prep_time=max_prep_time  # Add max preparation time filter
            )
        else:
            st.write("Using specific recipe search:", recipe)
            st.session_state["result"] = utils.find_similar_recipe(
                recipe, 
                st.session_state["data"], 
                num_recipes,
                min_calories=min_cal,
                max_calories=max_cal,
                meal_history=st.session_state["meal_history"]["recipes"],
                diabetic_friendly=st.session_state.get("has_diabetes", False),
                macro_targets=macro_targets,
                fitness_goal=fitness_goal,
                max_prep_time=max_prep_time  # Add max preparation time filter
            )
        
        if st.session_state["result"] is not None:
            st.write("Number of recommendations found:", len(st.session_state["result"]))
            if len(st.session_state["result"]) == 0:
                st.warning("No recipes found matching your criteria. Try adjusting your filters.")
        else:
            st.warning("No recommendations found. Try adjusting your filters or search criteria.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write("Data shape:", st.session_state["data"].shape)
        st.write("Data columns:", st.session_state["data"].columns.tolist())

search = st.button(label="Search", on_click=assign_values)

## Display the results
## -------------------------------------------------------------------

if search:
    if st.session_state["result"] is not None:
        # Show nutritional targets if they exist
        if st.session_state["user_info"]:
            st.subheader("Your Nutritional Targets")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Daily Calories", f"{st.session_state['user_info']['tdee']:.0f} kcal")
            with col2:
                st.metric("Protein", f"{st.session_state['user_info']['protein_target']:.0f}g")
            with col3:
                st.metric("Carbohydrates", f"{st.session_state['user_info']['carbs_target']:.0f}g")
            with col4:
                st.metric("Fats", f"{st.session_state['user_info']['fats_target']:.0f}g")
            st.markdown("---")
        
        for recipe_idx, row_index in enumerate(range(st.session_state["result"].shape[0])):
            dfx = st.session_state["result"].iloc[row_index]

            with st.expander(
                f"{recipe_idx+1}. "
                + f"{dfx['name'].capitalize()}"
            ):
                tab_1, tab_2, tab_3 = st.tabs(["Summary", "Ingredients", "Recipe"])

                with tab_1:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Calories", value=dfx["calories"])

                    with col2:
                        st.metric(label="Number of Steps", value=dfx["n_steps"])

                    with col3:
                        st.metric(label="Number of Ingredients", value=dfx["n_ingredients"])

                    with col4:
                        st.metric(label="Cooking Time", value=f"{dfx['minutes']} Mins")

                    # Show nutritional information
                    if st.session_state["user_info"]:
                        st.subheader("Nutritional Information (per serving)")
                        
                        try:
                            nutrition_cols = dfx.index[8:13]
                            
                            # Add validation for nutrition values
                            if any(dfx[nutrition_cols].isna()):
                                st.warning("Some nutritional information is missing for this recipe.")
                                continue
                            
                            # Convert PDV to grams using standard daily values
                            protein_g = max(0, dfx[nutrition_cols[3]] * 0.5)  # Ensure non-negative
                            carbs_g = max(0, dfx[nutrition_cols[1]] * 3)
                            fats_g = max(0, dfx[nutrition_cols[0]] * 0.65)
                            sugar_g = max(0, dfx[nutrition_cols[1]] * 0.5)
                            
                            # Display nutritional information
                            st.write("### Macronutrients")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Protein", f"{protein_g:.1f}g")
                            with col2:
                                st.metric("Carbs", f"{carbs_g:.1f}g")
                            with col3:
                                st.metric("Fats", f"{fats_g:.1f}g")
                            with col4:
                                st.metric("Sugar", f"{sugar_g:.1f}g")
                            
                            # Show warning only if user has diabetes and sugar content is high
                            if st.session_state["has_diabetes"] and sugar_g > 10:
                                st.warning(f"⚠️ High sugar content! This recipe contains {sugar_g:.1f}g of sugar per serving, which may not be suitable for diabetic patients.")
                            
                        except Exception as e:
                            st.error(f"Error calculating nutritional information: {str(e)}")
                            st.write("Please contact support if this error persists.")

                    fig = utils.plot_nutrition(dfx)
                    st.plotly_chart(fig)

                with tab_2:
                    st.text(f"Number of Ingredients: {dfx['n_ingredients']}")
                    for ingredient_idx, step in enumerate(dfx["ingredients"]):
                        st.markdown(f"{ingredient_idx+1}. {step}")

                with tab_3:
                    st.text(f"Recipe")
                    for step_idx, step in enumerate(dfx["steps"]):
                        st.markdown(f"{step_idx+1}. {step}")
                
                # Update the unique key to use recipe_idx
                if st.button(f"Add {dfx['name']} to Meal History", key=f"add_{dfx['name']}_{recipe_idx}"):
                    if dfx['name'] not in st.session_state["meal_history"]["recipes"]:
                        # Add recipe to history
                        st.session_state["meal_history"]["recipes"].append(dfx['name'])
                        
                        # Update last cooked date
                        st.session_state["meal_history"]["last_cooked"][dfx['name']] = datetime.now().strftime("%Y-%m-%d")
                        
                        # Update frequency
                        st.session_state["meal_history"]["frequency"][dfx['name']] = 1
                        
                        # Update preferences
                        # Extract cuisine from recipe name (simple implementation)
                        cuisine = dfx['name'].split()[0].lower()  # First word as cuisine
                        st.session_state["meal_history"]["preferences"]["favorite_cuisines"].add(cuisine)
                        
                        # Add ingredients to preferences
                        for ingredient in dfx['ingredients']:
                            st.session_state["meal_history"]["preferences"]["favorite_ingredients"].add(ingredient.lower())
                        
                        # Update preferred cooking time (running average)
                        current_avg = st.session_state["meal_history"]["preferences"]["preferred_cooking_time"]
                        n_recipes = len(st.session_state["meal_history"]["recipes"])
                        new_avg = (current_avg * (n_recipes - 1) + dfx['minutes']) / n_recipes
                        st.session_state["meal_history"]["preferences"]["preferred_cooking_time"] = new_avg
                        
                        # Update preferred meal size based on calories
                        current_avg = st.session_state["meal_history"]["preferences"]["preferred_meal_size"]
                        new_avg = (current_avg * (n_recipes - 1) + dfx['calories']) / n_recipes
                        st.session_state["meal_history"]["preferences"]["preferred_meal_size"] = new_avg
                        
                        st.experimental_rerun()
    else:
        st.warning("No recommendations found. Try adjusting your calorie range or adding more recipes to your history.")

