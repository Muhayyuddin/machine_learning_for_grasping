from typing import List, Any

from Processing import *
from Models import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data = read_dataset()
    data_shape(data)
    encoded_dataset, object_encoder, action_encoder, label_encoder = label_data(data)
    train, test = split_dataset(encoded_dataset)
    print("Train dataset shape : ")
    data_shape(train)
    print("test dataset shape : ")
    data_shape(test)
    check_balance(train)
    check_balance(test)
    describe_dataset(train)
    describe_dataset(test)

    train_x, train_y = extract_feature_values(train)
    test_x, test_y = extract_feature_values(test)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)

    bnb = train_berNB(train_x, train_y)
    dtc = train_DTC(train_x, train_y)
    gbc = train_GBC(train_x, train_y)
    gnb = train_GNB(train_x, train_y)
    knc = train_KNC(train_x, train_y)
    lda = train_LDA(train_x, train_y)
    lrc = train_LRC(train_x, train_y)
    rfc = train_RFC(train_x, train_y)
    lsv = train_LSVC(train_x, train_y)
    accuracy_list: List[Any] = [accuracy_score(bnb, test_x, test_y), accuracy_score(dtc, test_x, test_y),
                                accuracy_score(gbc, test_x, test_y), accuracy_score(gnb, test_x, test_y),
                                accuracy_score(knc, test_x, test_y), accuracy_score(lda, test_x, test_y),
                                accuracy_score(lrc, test_x, test_y), accuracy_score(rfc, test_x, test_y),
                                accuracy_score(lsv, test_x, test_y)]
    models_list: List[str] = ['BNB', "DTC", 'GBC', 'GNB', 'KNC', 'LDA', 'LRC', 'RFC', 'LSV']
    #    accuracy_score(lr, test_x, test_y)
    print(accuracy_list, len(accuracy_list), "accuracy_list")
    print(models_list, len(models_list), "models_list")
    predict_model_accuracy = pd.DataFrame({
        'Model': models_list,
        'Accuracy': accuracy_list
    })
    initialize_confusion_graph('bnb', bnb, test, label_encoder)
    initialize_confusion_graph('dtc', dtc, test, label_encoder)
    initialize_confusion_graph('gbc', gbc, test, label_encoder)
    initialize_confusion_graph('gnb', gnb, test, label_encoder)
    initialize_confusion_graph('knc', knc, test, label_encoder)
    initialize_confusion_graph('lda', lda, test, label_encoder)
    # initialize_confusion_graph('lr', lr,  test, label_encoder)
    initialize_confusion_graph('lrc', lrc, test, label_encoder)
    initialize_confusion_graph('rfc', rfc, test, label_encoder)
    initialize_confusion_graph('lsv', rfc, test, label_encoder)

    draw_predicted_graph(predict_model_accuracy)
