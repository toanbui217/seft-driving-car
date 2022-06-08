from model import *
import config as c


def main():
    # read training data to dataframe
    df = load_data(data_dir=c.data_log_dir)
    # shuffle
    df = df.sample(frac=1)

    split_point = int(np.floor(c.ratio_split * len(df)))

    model = AutoPilot(c.model_params)
    model.fit(df, split_point)


if __name__ == '__main__':
    main()
