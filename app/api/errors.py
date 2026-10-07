from starlette.exceptions import HTTPException


class AppError(HTTPException):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(status_code=status, detail={"code": code, "message": message})
        self.status = status
        self.code = code
        self.message = message
