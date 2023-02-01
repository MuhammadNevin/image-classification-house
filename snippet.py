predictions_vgg = model2_vgg_load.predict(test_features)
pred_labels_vgg = ((predictions_vgg > threshold)+0).ravel()

