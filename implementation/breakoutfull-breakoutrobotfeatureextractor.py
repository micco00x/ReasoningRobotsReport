class BreakoutNRobotFeatureExtractor(BreakoutRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            Discrete(287),
            Discrete(157),
        ))

        self.prev_ballX = 0
        self.prev_ballY = 0
        self.prev_paddleX = 0
        self.still_image = True

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        self.still_image = not self.still_image
        if self.still_image:
            return (self.prev_ballX-self.prev_paddleX+143, self.prev_ballY)
        # Extract position of the paddle:
        paddle_img = input[189:193,8:152,:]
        gray = cv2.cvtColor(paddle_img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        min_distance = np.inf
        paddleX = self.prev_paddleX
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            pX = int(M["m10"] / M["m00"])
            if abs(self.prev_paddleX - pX) < min_distance:
                min_distance = abs(self.prev_paddleX - pX)
                paddleX = pX

        # Extract position of the ball:
        ballX = self.prev_ballX
        ballY = self.prev_ballY
        ballspace_img = input[32:189,8:152,:]
        lower = np.array([200, 72, 72], dtype=np.uint8)
        upper = np.array([200, 72, 72], dtype=np.uint8)
        mask = cv2.inRange(ballspace_img, lower, upper)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            M = cv2.moments(c)
            # Avoid to compute position of the ball if M["m00"] is zero:
            if M["m00"] == 0:
                continue
            # Calculate the centroid
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Check that the centroid is actually part of the ball:
            left_black = False
            right_black = False
            if cX > 3:
                if ballspace_img[cY][cX-3][0] != 200 or \
                    ballspace_img[cY][cX-3][1] != 72 or \
                    ballspace_img[cY][cX-3][2] != 72:
                    left_black = True
            else:
                 if ballspace_img[cY][cX+3][0] != 200 or \
                     ballspace_img[cY][cX+3][1] != 72 or \
                     ballspace_img[cY][cX+3][2] != 72:
                     right_black = True
            if left_black or right_black:
                ballX = cX
                ballY = cY

        self.prev_ballX = ballX
        self.prev_ballY = ballY
        self.prev_paddleX = paddleX

        return (self.prev_ballX - self.prev_paddleX + 143, self.prev_ballY)
