import numpy as np


'''
 APEX Algorithm for adaptive Principal Component Analysis
'''
def fit_apex(x, MAX_COMPONENTS, MAX_EPOCHS=100, verbose=False):
    # x : data matrix. Shape = NUM_PATTERNS x DIMENSIONS
    NUM_PATTERNS = x.shape[0]
    DIM = x.shape[1]
    # Remove the mean from each pattern
    x = x - np.tile(x.mean(axis=0), (NUM_PATTERNS,1))
    # Starting learning rate
    beta = 0.00001
    # Initialize component matrix
    A = []
    
    for component in range(1,MAX_COMPONENTS+1):
        w = np.random.randn(DIM)  #Initialize component
        if component > 1:
            c = np.random.randn(component-1)  #Initialize lateral weights
        
        for epoch in range(1,MAX_EPOCHS+1):
            s = 0
            
            for i in range(0,NUM_PATTERNS):
                if component > 1:
                    yprev = A.dot(x[i])
                    y = np.inner(w,x[i]) - np.inner(c,yprev)
                    c = c + beta * (yprev*y - c*y*y)
                else:
                    y = np.inner(w,x[i])
                w = w + beta * (x[i]*y - w*y*y)
                s += y*y
            
            if verbose:
                print "Epoch:{0}, ||w||={1}".format(epoch, np.linalg.norm(w))
            # Update learning rate
            beta = 0.1/s
        
        if component == 1:
            w = w / np.linalg.norm(w)
            A = w.reshape(1,DIM)
        else:
            w = w - c.dot(A)
            w = w / np.linalg.norm(w)
            A = np.vstack((A,w))
    
    x_transformed = np.dot(x, A.T)
    return A, x_transformed
    
