import cv2
import random
import numpy as np
import time
import Queue
import threading
import globals as g_
from concurrent.futures import ThreadPoolExecutor

W = H = 256

class Shape:
    def __init__(self, list_file):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]
        
        self.views = self._load_views(view_files, self.V)
        self.done_mean = False
        

    def _load_views(self, view_files, V):
        views = []
        for f in view_files:
            im = cv2.imread(f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            assert im.shape == (W,H,3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)
        return views
    
    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.views[:,:,:,i] -= mean_bgr[i]
            
            self.done_mean = True
    
    def crop_center(self, size=(227,227)):
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        left = w / 2 - wn / 2
        top = h / 2 - hn / 2
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]
    
    
class Dataset:
    def __init__(self, listfiles, labels, subtract_mean, V):
        self.listfiles = listfiles
        self.labels = labels
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V
        print 'dataset inited'
        print '  total size:', len(listfiles)
    
    def shuffle(self):
        z = zip(self.listfiles, self.labels)
        random.shuffle(z)
        self.listfiles, self.labels = [list(l) for l in zip(*z)]
        self.shuffled = True


    def batches(self, batch_size):
        for x,y in self._batches_fast(self.listfiles, batch_size):
            yield x,y
        
    def sample_batches(self, batch_size, n):
        listfiles = random.sample(self.listfiles, n)
        for x,y in self._batches_fast(listfiles, batch_size):
            yield x,y

    def _batches(self, listfiles, batch_size):
        n = len(listfiles)
        for i in xrange(0, n, batch_size):
            starttime = time.time()

            lists = listfiles[i : i+batch_size]
            x = np.zeros((batch_size, self.V, 227, 227, 3)) 
            y = np.zeros(batch_size)
            
            for j,l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                x[j, ...] = s.views
                y[j] = s.label
            
            print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y
    
    def _load_shape(self, listfile):
        s = Shape(listfile)
        s.crop_center()
        if self.subtract_mean:
            s.subtract_mean()
        return s 

    def _batches_fast(self, listfiles, batch_size):
        subtract_mean = self.subtract_mean
        n = len(listfiles)

        def load(listfiles, q, batch_size):
            n = len(listfiles)
            with ThreadPoolExecutor(max_workers=16) as pool:
                for i in range(0, n, batch_size):
                    sub = listfiles[i: i + batch_size] if i < n-1 else [listfiles[-1]]
                    shapes = list(pool.map(self._load_shape, sub))
                    views = np.array([s.views for s in shapes])
                    labels = [s.label for s in shapes]
                    q.put((views, labels))

            # indicate that I'm done
            q.put(None)

        # This must be larger than twice the batch_size
        q = Queue.Queue(maxsize=g_.INPUT_QUEUE_SIZE)

        # background loading Shapes process
        p = threading.Thread(target=load, args=(listfiles, q, batch_size))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()


        x = np.zeros((batch_size, self.V, 227, 227, 3)) 
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()
            
            item = q.get()
            if item is None:
                break
            x, y = item

            # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.listfiles)


