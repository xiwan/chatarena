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
    env_conf = 'whoisthespy'
    app.arena = Arena.from_config(f'examples/{env_conf}.json')
    return jsonify({'msg': f"arena {env_conf} is loaded!"})

@app.route('/chatarena/reset', methods=['POST'])
def reset_game():
    if app.arena is None:
        return jsonify({'error': "arena is undefined!"}), 400
    app.arena.reset()
    app.arena = None
    return jsonify({'msg': "arena is unloaded!"})
    
#curl -X POST http://localhost:8080/chatarena/step
@app.route('/chatarena/step', methods=['POST'])
def step_game():
    if app.arena is None:
        return jsonify({'error': "arena is undefined!"}), 400

    try:
        data = request.get_json()
        i_player_name = data.get('player_name', "")
        i_player_action = data.get('player_action', "")
        
        timestep = app.arena.step(i_player_name, i_player_action)
        observation = timestep.observation
        terminal = timestep.terminal
        # if timestep.terminal:
        #     raise Exception("arena is end!")
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
