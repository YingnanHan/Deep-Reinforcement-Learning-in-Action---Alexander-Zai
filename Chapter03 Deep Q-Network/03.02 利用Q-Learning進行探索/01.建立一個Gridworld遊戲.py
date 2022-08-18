from Gridworld import Gridworld

if __name__ == '__main__':
    game = Gridworld(size=4, mode='static')
    print(game.display(),"\n")
    game.makeMove('d')
    print(game.display(),"\n")
    print(game.reward())
    print(game.board.render_np()) # 顯示遊戲的當前狀態
    print(game.board.render_np().shape) # 輸出張量狀態的shape