import json
from flask import Flask, request, jsonify
from chatarena.arena import Arena

app = Flask(__name__)

#curl -X POST http://localhost:8080/chatarena/step
@app.route('/chatarena/step', methods=['POST'])
def step_game():

    # 创建Arena对象
    arena = Arena.from_config('examples/chameleon.json')

    try:
        timestep = arena.step()
        observation = timestep.observation
        terminal = timestep.terminal
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # 返回结果
    response = {
        'observation': observation,
        'terminal': terminal
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
