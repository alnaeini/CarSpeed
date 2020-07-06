import cv2
import os
import sys
import numpy as np
from sklearn import linear_model
import queue
from tools import movingAverage, plot, computeAverage
import matplotlib.pyplot as plt


class Speed_Car():
    def __init__(self, video_train_path, text_train_path, video_test_path):
        # Train video and text path
        self.v_train = cv2.VideoCapture(video_train_path)
        self.t_train = text_train_path
        # Number of frames for 17 min o f 20 fps video
        self.n_frames = 17*60*20
        # Read test video
        self.test_vid = cv2.VideoCapture(video_test_path)
        # Generate test.txt
        self.predict = True
        # Generate visualization
        self.visual = False
        # Separate function to allow for different methods to be inculcated into the same class
        self.parameters()
        # test text directory
        self.t_text = True
        # See the camera in the test
        self.camera = False 

    def parameters(self):
        """ Extract parameters for the Lucas-Kanade method  """

        # Using Lucas-Kanade method to estimate the optical flow
        self.lkparameter = dict(winSize=(21, 21),
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

        self.frame_idx = 0
        self.prev_pts = None
        self.detect_interval = 1
        self.temp_preds = np.zeros(int(self.v_train.get(cv2.CAP_PROP_FRAME_COUNT)))

        """ load traning text file """
        with open(self.t_train, 'r') as file_:
            gt = file_.readlines()
            gt = [float(x.strip()) for x in gt]

        self.gt = np.array(gt[:self.n_frames])
        self.window = 80  # for moving average
        self.prev_gray = None

    def focus(self, mask=None, test=False):
        """ Focus on the road """

        vid = self.test_vid if test else self.v_train

        if mask is None:
            W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            mask = np.zeros(shape=(H, W), dtype=np.uint8)
            mask.fill(255)
        else:
            W = mask.shape[1]
            H = mask.shape[0]

        cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)

        x_top_offset = 240
        x_btm_offset = 65

        poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250],
                              [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
        cv2.fillPoly(mask, poly_pts, (255, 255, 255))

        return mask

    def opticalflow(self, frame):
        """ calculating optical flow """
        # blur the surrondings
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # Store Flow (x, y, dx, dy)
        curr_pts, _st, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame, self.prev_pts, None, **self.lkparameter)

        # Store Flow(x, y, dx, dy)
        flow = np.hstack((self.prev_pts.reshape(-1, 2),
                          (curr_pts - self.prev_pts).reshape(-1, 2)))

        preds = []

        for x, y, u, v in flow:
            if v < -0.05:
                continue
            # Translate points to center

            x -= frame.shape[1]/2
            y -= frame.shape[0]/2

            # Append Preds taking care of stability issues
            if y == 0 or (abs(u) - abs(v)) > 11:
                preds.append(0)
                preds.append(0)
            elif x == 0:
                preds.append(0)
                preds.append(v/y**2)
            else:
                preds.append(u/y**2)
                preds.append(v/y**2)

        return [n for n in preds if n >= 0]

    def KeyPts(self, offset_x=0, offset_y=0):
        """ return key points from  """

        if self.prev_pts is None:
            return None
        return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size=10) for p in self.prev_pts]

    def features(self, frame, mask):
        return cv2.goodFeaturesToTrack(frame, 30, 0.1, 10, blockSize=10, mask=mask)

    def run(self):
        # Construct mask first
        mask = self.focus()
        prev_key_pts = None

        while self.v_train.isOpened() and self.frame_idx < len(self.gt):
            ret, frame = self.v_train.read()
            if not ret:
                break

            # Convert to B/W
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray[130:350, 35:605]
            mask_vis = frame.copy()  # visualiatiob

            # Process each frame
            if self.prev_pts is None:
                self.temp_preds[self.frame_idx] = 0
            else:
                # Get mediun of V/hf values
                preds = self.opticalflow(frame_gray)
                self.temp_preds[self.frame_idx] = np.median(
                    preds) if len(preds) else 0

            # Extract features
            self.prev_pts = self.features(frame_gray, mask[130:350, 35:605])
            self.prev_gray = frame_gray
            self.frame_idx += 1

            # for Visualization purpose only
            if self.visual:
                prev_key_pts = self.visualize(frame, mask_vis, prev_key_pts)
                if cv2.waitKey(1) and 0xFF == ord('q'):
                    break

        # self.video.release()
        self.v_train.release()

        # split train mp4 to train and validation
        split = self.frame_idx//10
        train_preds = self.temp_preds[:self.frame_idx-split]
        val_preds = self.temp_preds[self.frame_idx - split:self.frame_idx]
        gt_train = self.gt[:len(train_preds)]
        gt_val = self.gt[len(train_preds):self.frame_idx]

        # fit to ground truth (moving average)
        preds = movingAverage(train_preds, self.window)
        lin_reg = linear_model.LinearRegression(fit_intercept=False)
        lin_reg.fit(preds.reshape(-1, 1), gt_train)
        hf_factor = lin_reg.coef_[0]
        print("Estimated hf factor = {}".format(hf_factor))

        # estimate training error
        pred_speed_train = train_preds * hf_factor
        pred_speed_train = movingAverage(pred_speed_train, self.window)
        self.mse_train = np.mean((pred_speed_train - gt_train)**2)
        print("Mean Squared Error for train dataset", self.mse_train)

        # estimate validation error
        pred_speed_val = val_preds * hf_factor
        pred_speed_val = movingAverage(pred_speed_val, self.window)
        self.mse_test = np.mean((pred_speed_val - gt_val)**2)
        print("Mean Squared Error for validation dataset", self.mse_test)

        return hf_factor

    def visualize(self, frame, mask_vis, prev_key_pts, speed=None):
        self.focus(mask_vis)
        mask_vis = cv2.bitwise_not(mask_vis)
        frame_vis = cv2.addWeighted(frame, 1, mask_vis, 0.3, 0)
        key_pts = self.KeyPts(35, 130)
        cv2.drawKeypoints(frame_vis, key_pts, frame_vis, color=(0, 0, 255))
        cv2.drawKeypoints(frame_vis, prev_key_pts,
                          frame_vis, color=(0, 255, 0))

        if speed:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_vis, "speed {}".format(speed),
                        (10, 35), font, 1.2, (0, 0, 255))
        cv2.imshow('test', frame_vis)

        return key_pts

    def test(self, hf_factor, save_txt=True):
        mask = self.focus(test=True)

        self.prev_gray = None
        test_preds = np.zeros(int(self.test_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_idx = 0
        frame_index =[]
        curr_estimate = 0
        prev_key_pts = None
        self.prev_pts = None

        while self.test_vid.isOpened():
            ret, frame = self.test_vid.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray[130:350, 35:605]

            # process each frame

            prev_speed = 0
            if self.prev_pts is None:
                test_preds[frame_idx] = 0
            else:
                # get median of predicted V/hf values
                preds = self.opticalflow(frame_gray)
                prev_speed = np.median(preds) * hf_factor if len(preds) else 0
                test_preds[frame_idx] = prev_speed

            # Extract features
            self.prev_pts = self.features(frame_gray, mask[130:350, 35:605])
            self.prev_gray = frame_gray
            frame_idx += 1
            frame_index.append(frame_idx)

            # for visulization
            mask_vis = frame.copy()  # <- For visualization
            vis_pred_speed = computeAverage(
                test_preds, self.window//2, frame_idx)
            if self.camera:
                prev_key_pts = self.visualize(
                    frame, mask_vis, prev_key_pts, speed=vis_pred_speed)

        if self.predict:
            with open("test.txt", "w") as file_:
                for item in test_preds:
                    file_.write("%s \n" % item)
        print('predictions value are successfully saved in the text.txt file in the current directory')


if __name__ == '__main__':
    video_train_path = 'data/train.mp4'
    text_train_path = 'data/train.txt'
    video_test_path = 'data/test.mp4'
    speedcar = Speed_Car(video_train_path, text_train_path, video_test_path)
    hf = speedcar.run()
    speedcar.test(hf, True)
    print(speedcar.mse_train)
    print(speedcar.mse_test)
    cv2.destroyAllWindows()
