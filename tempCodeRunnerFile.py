    #visual 2D display of the petal lenght according to the sepal width
    plt.figure()
    plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c = data.target)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Iris dataset')
    plt.show()