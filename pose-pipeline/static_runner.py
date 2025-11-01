import subprocess
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def process_file():
    file_name = request.json.get('file_name')
    basename = os.path.basename(file_name)

    file_name = f'in/{basename}.mp4'

    if not file_name or not file_name.endswith('.mp4'):
        return jsonify({'error': 'Invalid file name. File must end with .mp4'}), 400

    try:
        # Run the echo command using subprocess
        result = subprocess.run(['python3', 'main.py', 'hands', file_name, f'{basename}.h5'], capture_output=True, text=True)
        
        # Return the output from the echo command
        return jsonify({'message': result.stdout.strip()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=80)