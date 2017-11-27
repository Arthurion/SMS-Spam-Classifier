def divide_if(X, y, filter):
	X1 = list(X[k] for k in range(0,len(X)) if filter in X[k])
	X2 = list(k for k in X if k not in X1)
	y1 = list(y[k] for k in range(0,len(y)) if filter in X[k])
	y2 = list(k for k in y if k not in y1)
	return (X1,X2,y1,y2)

def divide_if_one(X, y):
	X1 = list(X[k] for k in range(0,len(X)) if y[k] == 1)
	X2 = list(k for k in X if k not in X1)
	return (X1,X2)