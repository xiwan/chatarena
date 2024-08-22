import json
from flask import Flask, request, jsonify
from chatarena.arena import Arena

app = Flask(__name__)
app.arena = None

@app.route('/', methods=['GET'])
def health():
    return jsonify({'msg': "ok"})

@app.route('/chatarena/setup', methods=['POST'])
def setup_game():
    # 创建Arena对象
    app.arena = Arena.from_config('examples/chameleon.json')
    return jsonify({'msg': "arena is loaded!"})

#curl -X POST http://localhost:8080/chatarena/step
@app.route('/chatarena/step', methods=['POST'])
def step_game():
    if app.arena is None:
        return jsonify({'error': "arena is undefined!"}), 400

    try:
        timestep = app.arena.step()
        if timestep.terminal:
            raise Exception("arena is end!")
            
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
    app.run(debug=True, host='0.0.0.0', port=8080)
