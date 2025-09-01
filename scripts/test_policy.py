import onnxruntime as ort

session = ort.InferenceSession("legged_gym/logs/h1:teleop/exported/policies/25_08_29_01-19-54_OmniH2O_STUDENTmodel_20000.onnx")
print("Inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])
