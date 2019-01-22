def hitDetect(self):
    ##COLLISION DETECTION

    # [...]

    for brick in self.bricks:
        if brick.rect.colliderect(ball_rect):
            if ((not self.se_brick is None)):
                self.se_brick.play()

            self.score = self.score + 1
            self.brick_hit_count += 1
            self.bricks.remove(brick)
            self.last_brikcsremoved.append(brick)
            self.bricksgrid[(brick.i, brick.j)] = 0

            # bug correction begin

            min_distance = math.fabs(self.ball_x - brick.x)
            edge_hit = 0
            tmp = math.fabs(self.ball_x - (brick.x + block_width))
            if tmp<min_distance:
                min_distance = tmp
                edge_hit = 1
            tmp = math.fabs(self.ball_y - brick.y)
            if tmp<min_distance:
                min_distance = tmp
                edge_hit = 2
            tmp = math.fabs(self.ball_y - (brick.y + block_height))
            if tmp<=min_distance:
                edge_hit = 3

            if edge_hit == 0: #left edge
                if self.ball_speed_x > 0:
                    self.ball_speed_x = -self.ball_speed_x
            elif edge_hit == 1: #right edge
                if self.ball_speed_x < 0:
                    self.ball_speed_x = -self.ball_speed_x
            elif edge_hit == 2: #top edge
                if self.ball_speed_y > 0:
                    self.ball_speed_y = -self.ball_speed_y
            else: #bottom edge
                if self.ball_speed_y < 0:
                    self.ball_speed_y = -self.ball_speed_y

            # bug correction end

            self.current_reward += self.STATES['Scores']
            self.paddle_hit_without_brick = 0
            break