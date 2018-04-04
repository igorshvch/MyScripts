import matplotlib.pyplot as plt

def sp(sizes, labels=None):
    #assert len(labels) == len(sizes), 'Not equal!'
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.show()