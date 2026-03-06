def setup():
    """触发模块导入，注册任务类型"""
    from . import invoke_agent, send_static_message  # noqa: F401
