from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2):
    """
    Membagi data menjadi data training dan testing (default 80:20).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # Jangan diacak (time series)
    )
    return X_train, X_test, y_train, y_test
