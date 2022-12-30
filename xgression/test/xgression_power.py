import pygame as pg
pg.init()
Font=pg.font.SysFont('timesnewroman',  15)
clock = pg.time.Clock()
screen = pg.display.set_mode((400,400))
screen.fill("black")
running = True

class ClickHandler:
    released = True
    mouse_position = (0,0)
    def single_click(self) -> bool:
        mouse_clicked = pg.mouse.get_pressed()[0]
        if mouse_clicked:
            if self.released:
                self.released = False
                self.mouse_position = pg.mouse.get_pos()
                return True
        else:
            self.released = True
        return False

class DataCollectorScreen:
    def __init__(self) -> None:
        self.value = []

        self.y_min = 0
        self.y_max = 10
        
        self.x_min = 0
        self.x_max = 10

        self.screen_size_x = screen.get_width()
        self.screen_size_y = screen.get_height()

    def dotting_handler(self):
        if click_handler.single_click():
            mouse_pos = click_handler.mouse_position
            mouse_pos[0] = round(0)
            pg.draw.circle(screen, "red", mouse_pos, 5)
            #screen.blit(letter1, (662-i, -162+i))

            # convert viewport to relative cordinate to domain and range
            x = ((((mouse_pos[0]/self.screen_size_x))*(self.x_max-self.x_min))+self.x_min)
            y = self.y_max - ((((mouse_pos[1]/self.screen_size_y)-self.y_min)*(self.y_max-self.y_min))+self.y_min)
            cordinate_text = Font.render(f'({x}, {y})',True,"white")

            screen.blit(cordinate_text, mouse_pos)

        

    

click_handler = ClickHandler()
d1 = DataCollectorScreen()
while running:
    for event in pg.event.get():
        # Handle quitting the program
        if event.type == pg.QUIT:
            running = False

    #if click_handler.single_click():
    #    print(click_handler.mouse_position)

    d1.dotting_handler()
    pg.display.flip()
    clock.tick(30)


pg.quit()