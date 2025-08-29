from fuzzy_triage import triage_priority, pain, heart_rate, breathing

def get_priority(pain_level, heart_rate_level, breathing_level):
    # Run the fuzzy simulation
    output = triage_priority.simulate({
        'pain': pain_level,
        'heart_rate': heart_rate_level,
        'breathing': breathing_level
    })
    return output['priority']

if __name__ == "__main__":
    print("🏥 Simple Hospital Triage System 🏥")
    
    pain_level = float(input("Enter pain level (0-10): "))
    heart_rate_level = float(input("Enter heart rate (40-180): "))
    breathing_level = float(input("Enter breathing rate (5-40): "))
    
    priority = get_priority(pain_level, heart_rate_level, breathing_level)
    print(f"\nPatient Priority (0-100): {priority:.2f}")
    
    if priority < 30:
        print("➡️ Low Priority – Can Wait")
    elif 30 <= priority < 70:
        print("➡️ Medium Priority – Needs Attention Soon")
    else:
        print("➡️ High Priority – Immediate Attention Required!")
