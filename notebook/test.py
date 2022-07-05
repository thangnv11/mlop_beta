from IPython.core.display import display

from mlop_beta.dataset_calling import load_data


if __name__ == '__main__':
    dt = load_data()
    display(dt)