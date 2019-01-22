def getstate(self):
    resx = b.resolutionx  # highest resolution
    resy = b.resolutiony  # highest resolution
    if (self.ball_y < self.win_height // 3):  # upper part, lower resolution
        resx *= 3
        resy *= 3
    elif (self.ball_y < 2 * self.win_height // 3):  # lower part, medium resolution
        resx *= 2
        resy *= 2

    ball_x = int(self.ball_x) // resx
    ball_y = int(self.ball_y) // resy
    ball_dir = 0
    if self.ball_speed_y > 0:  # down
        ball_dir += 5
    if self.ball_speed_x < -2.5:  # quick-left
        ball_dir += 1
    elif self.ball_speed_x < 0:  # left
        ball_dir += 2
    elif self.ball_speed_x > 2.5:  # quick-right
        ball_dir += 3
    elif self.ball_speed_x > 0:  # right
        ball_dir += 4

    if self.simple_state:
        paddle_x = 0
    else:
        paddle_x = int(self.paddle_x) // resx

    diff_paddle_ball = int((self.ball_x - self.paddle_x + self.win_width) / b.resolutionx)

    return {
        "ball_x":   ball_x,
        "ball_y":   ball_y,
        "ball_dir": ball_dir,
        "paddle_x": paddle_x,
        "diff_paddle_ball": diff_paddle_ball,
        "bricks_matrix": self.bricksgrid
    }