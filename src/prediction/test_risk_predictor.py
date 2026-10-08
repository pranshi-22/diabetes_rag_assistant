from risk_predictor import predict_diabetes_risk


def main():

    print("=" * 60)
    print("DIABETES RISK ASSESSMENT TEST")
    print("=" * 60)

    result = predict_diabetes_risk(
        gender="Female",
        age=45,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=27.5,
        HbA1c_level=5.8,
        blood_glucose_level=120,
    )

    print("\nRISK ASSESSMENT")
    print("-" * 60)

    print(f"Risk probability : {result['risk_probability']}")
    print(f"Risk percentage  : {result['risk_percentage']}%")
    print(f"Risk level       : {result['risk_level']}")

    print("\nMESSAGE")
    print("-" * 60)
    print(result["message"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()