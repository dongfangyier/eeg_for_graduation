import all_analyze.all_in_one as all_inone
import all_analyze.sklearn_select_features_all_inone as select_features
import all_analyze.features_classify as classify


if __name__ == "__main__":
    print("all in one ...")
    all_inone.start()
    print("select features ...")
    select_features.start()
    print("classify ...")
    classify.start()