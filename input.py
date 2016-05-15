import cv2

W = H = 256

class Shape:
    def __init__(self, list_file):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]
        
        self.views = _load_views(self.view_files, self.V)
        self.done_mean = False
        

    def _load_views(view_files, V):
        views = []
        for f in view_files:
            im = cv2.imread(f)
            im = cv2.resize((W, H))
            im = cv2.cvtColor(cv2.COLOR_GRAY2BGR) #BGR!!
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
        views = views[:, left:right, top:bottom, :]
    
    
class Dataset:
    def __init__(self, listfiles, subtract_mean, V):
        self.listfiles = listfiles
        self.listfiles_val = []
        self.shuffled = False
        self.splitted = False
        self.subtract_mean = subtract_mean
        self.V = V
    
    def shuffle(self):
        self.listfiles = random.shuffle(self.listfiles)
        self.shuffled = True

    def split_val(self, split=(9,1)):
        if not self.shuffled:
            print 'Warning: not shuffled!'
        if not self.splitted:
            n = len(self.listfiles)
            cut = n * split[0] / sum(split)
            self.listfiles_val = self.listfiles[cut:]
            self.listfiles_val = self.listfiles[:cut]
            self.splitted = True 

    def batches(self, batch_size):
        for x,y in self._batches(listfiles, batch_size)
            yield x,y
        

    def validation_batch(self, batch_size):
        assert self.splitted, 'Not splitted to train/val!'
        for x,y in self._batches(listfiles_val, batch_size)
            yield x,y

    def _batches(self, listfiles, batch_size):
        n = len(listfiles)
        for i in xrange(0, n, batch_size):
            lists = listfiles[i : i+batch_size]
            x = np.zeros((batch_size, self.V, W, H, 3)) 
            y = np.zeros(batch_size)
            
            for j,l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                x[j, ...] = s.views
                y[j] = s.label
            
            yield x, y
    



