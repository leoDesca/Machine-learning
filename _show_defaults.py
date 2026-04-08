from preprocess import DEFAULT_PROFILE
keys=['Period_Code','Is_Weekend','Is_Exam_Period','Is_Break','Top_Meal_Share','Meal_Entropy','Sold_Roll7_Mean','Sold_Roll7_Std','Waste_Roll7_Mean','Profit_Roll7_Mean']
for k in keys:
    print(k, DEFAULT_PROFILE.get(k))
