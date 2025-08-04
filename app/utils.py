def predict_diabetes(model, inputs):
    return model.predict([inputs])[0]

def predict_heart_disease(model, inputs):
    return model.predict([inputs])[0]
