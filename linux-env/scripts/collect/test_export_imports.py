"""Standalone exports must resolve pure IK without a sourced ROS workspace."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path

class StandaloneExportTests(unittest.TestCase):
    def test_export_imports_own_pure_control_package_with_ros_environment_cleared(self):
        here=Path(__file__).resolve().parent;repo=here.parents[2]
        code='''import sys,runpy,pathlib
sys.path.insert(0,sys.argv[1])
runpy.run_path(str(pathlib.Path(sys.argv[1])/'export.py'),run_name='standalone_export_probe')
from so_arm101_control import grasp_geometry,compute_workspace
assert pathlib.Path(grasp_geometry.__file__).resolve()==pathlib.Path(sys.argv[2])/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/grasp_geometry.py'
assert 'rclpy' not in sys.modules and 'tkinter' not in sys.modules
print('Pure calibrated IK available without ROS or Tk')
'''
        env={k:v for k,v in os.environ.items() if not (k.startswith(('ROS_','AMENT_','COLCON_')) or k in ('PYTHONPATH','LD_LIBRARY_PATH','CMAKE_PREFIX_PATH'))}
        env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as cwd:
            result=subprocess.run([sys.executable,'-B','-c',code,str(here),str(repo)],cwd=cwd,env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('without ROS or Tk',result.stdout)

if __name__=='__main__':unittest.main()
