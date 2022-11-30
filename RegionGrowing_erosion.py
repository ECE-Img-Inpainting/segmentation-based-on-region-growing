import cv2
import numpy as np
import matplotlib.pyplot as plt
import queue

k = 50

def neighbor(img,x,y):
    H, W = img[:,:,0].shape
    l = []
    if x+1 < H:
        l.append([x+1,y])
    if x-1 >= 0:
        l.append([x-1,y])
    if y+1 < W:
        l.append([x,y+1])
    if y-1 >= 0:
        l.append([x,y-1])
    return l

def difference(a,b):
    return np.linalg.norm(a-b)


# img = np.array([[10,10,10,10,10,10,10],[10,10,10,69,70,10,10],[59,10,60,64,59,56,60],
# [10,59,10,60,70,10,62],[10,60,59,65,67,10,65],[10,10,10,10,10,10,10],[10,10,10,10,10,10,10]])
img = cv2.imread('/Users/zhaosonglin/Desktop/project example/fence.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

plt.imshow(img)
y0, x0 = plt.ginput(1, timeout=0)[0]
seed = [int(x0), int(y0)]
# seed = [3,4]
q = queue.Queue()
q.put(seed)
visited = np.zeros_like(img[:,:,0])
mask = np.zeros_like(img[:,:,0])
visited[seed[0],seed[1]] = 1
mask[seed[0],seed[1]] = 1
print(visited)
while not q.empty():
    point = q.get()
    x,y = point
    # avg = np.mean(img[np.where(visited == 1)])
    avg1 = np.mean(img[:,:,0][np.where(mask == 1)])
    avg2 = np.mean(img[:,:,1][np.where(mask == 1)])
    avg3 = np.mean(img[:,:,2][np.where(mask == 1)])
    avg = np.array([avg1,avg2,avg3])
    print(avg)
    for p in neighbor(img, x, y):
        a,b = p
        visited[a][b] = 1
        if mask[a][b] == 0 and difference(img[a][b],avg) <= k:
            mask[a,b] = 1
            q.put(p)


print(mask)
plt.show()

plt.imshow(mask,'gray')
plt.show()
