import os
import subprocess
from core.context import Context

class LayoutModule:
    """
    Module: Scene Layout Optimization (Steps 9-11)
    模块：场景布局优化 (步骤 9-11)
    
    Wraps Blender Script execution.
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S4_blender_layout_and_corr')
        
    def run(self):
        self.logger.info(">>> Stage 5: Layout Optimization (Blender)")
        
        # Get placement info path from previous step or construct it
        placement_json_path = self.context.get_data('placement_info_path')
        
        if not placement_json_path:
            # Fallback logic
            scene_name = self.context.image_name
            S3_folder = os.path.join(self.context.output_dir, 'S3_pose_inference')
            placement_json_path = os.path.join(S3_folder, f'{scene_name}_placement_info.json')
            
        if not os.path.exists(placement_json_path):
            raise FileNotFoundError(f"Placement info not found: {placement_json_path}")
        
        # Create S4 output folder
        S4_folder = os.path.join(self.context.output_dir, 'S4_layout_refinement')
        os.makedirs(S4_folder, exist_ok=True)
        
        # Smart resume: Check if S4 output files already exist in the new folder
        scene_name = self.context.image_name
        s4_json_path = os.path.join(S4_folder, f'{scene_name}_placement_info_s4.json')
        s4_render_path = os.path.join(S4_folder, f'{scene_name}_render_simu.png')
        
        if not self.context.clean_mode:
            if os.path.exists(s4_json_path) and os.path.exists(s4_render_path):
                self.logger.info(f"✓ S4 已完成：所有必需文件都存在，跳过此阶段")
                self.logger.info(f"  - {os.path.basename(s4_json_path)}: ✓")
                self.logger.info(f"  - {os.path.basename(s4_render_path)}: ✓")
                self.logger.info("Layout Optimization Done (Skipped, final results exist).")
                return
            
        # Path to the new blender script
        script_path = "modules/S4_blender_layout_and_corr.py"
        
        # Ensure models are released before running Blender to free VRAM
        self.context.release_models()
        
        # Blender Command
        blender_cmd = [
            "blender",
            "--background",
            "--python", script_path,
            "--",
            "--obj_placement_info_json_path", placement_json_path,
            "--output_folder", S4_folder
        ]
        
        # 添加debug参数（如果启用）
        if self.context.debug_mode:
            blender_cmd.append("--debug")
        
        self.logger.info(f"Executing Blender: {' '.join(blender_cmd)}")
        self.logger.info("⏳ 正在执行Blender摆放、逻辑优化和掉落仿真，这可能需要几分钟时间，请耐心等待...")
        
        try:
            import time
            import threading
            
            process = subprocess.Popen(
                blender_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # 行缓冲
                cwd=os.getcwd()
            )
            
            # 启动进度指示线程
            stop_indicator = threading.Event()
            def progress_indicator():
                count = 0
                while not stop_indicator.is_set():
                    count += 1
                    if count % 10 == 0:  # 每10秒输出一次
                        self.logger.info(f"⏳ Blender仍在运行... ({count}秒)")
                    time.sleep(1)
            
            progress_thread = threading.Thread(target=progress_indicator, daemon=True)
            progress_thread.start()
            
            try:
                # 读取输出
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        # 过滤掉一些Blender的噪音输出
                        if not any(x in line.lower() for x in ['found bundled python', 'read prefs']):
                            self.logger.info(f"[Blender] {line}")
                
                process.wait()
            finally:
                # 停止进度指示线程
                stop_indicator.set()
                progress_thread.join(timeout=1)
            
            if process.returncode != 0:
                self.logger.error("Blender process failed.")
                raise subprocess.CalledProcessError(process.returncode, blender_cmd)
                
        except Exception as e:
            self.logger.error(f"Layout Optimization Failed: {e}")
            raise e
            
        self.logger.info("Layout Optimization Done.")
        
        # Blender process naturally releases its own resources on exit, 
        # but we can ensure the pipeline context is clean if needed.
        if self.context.clean_mode:
            self.context.cleanup()

