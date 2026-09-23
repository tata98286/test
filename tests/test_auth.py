import secrets
import unittest

from app import app, get_db
from werkzeug.security import check_password_hash


class AuthFlowTest(unittest.TestCase):
    def test_registration_login_and_logout(self):
        app.config['TESTING'] = True
        client = app.test_client()
        username = 'test_' + secrets.token_hex(10)
        password = 'Test-password-123!'

        def post(path, **data):
            with client.session_transaction() as state:
                data['csrf_token'] = state['csrf_token']
            return client.post(path, data=data)

        try:
            self.assertEqual(client.get('/').status_code, 302)
            self.assertEqual(client.get('/signup').status_code, 200)
            self.assertEqual(client.post('/signup', data={}).status_code, 400)
            self.assertEqual(post('/signup', username=username, password=password, password_confirm='mismatch').status_code, 400)
            self.assertEqual(post('/signup', username=username, password=password, password_confirm=password, role='ADMIN').status_code, 302)
            with app.app_context():
                with get_db().cursor() as cursor:
                    cursor.execute('SELECT * FROM `user` WHERE username=%s', (username,))
                    user = cursor.fetchone()
            self.assertEqual(user['role'], 'USER')
            self.assertNotEqual(user['password_hash'], password)
            self.assertTrue(user['password_hash'].startswith('scrypt:'))
            self.assertTrue(check_password_hash(user['password_hash'], password))
            self.assertEqual(post('/signup', username=username, password=password, password_confirm=password).status_code, 400)
            self.assertEqual(post('/login', username=username, password='wrong').status_code, 401)
            self.assertEqual(post('/login', username="' OR 1=1 --", password=password).status_code, 401)
            self.assertEqual(post('/login', username=username, password=password).status_code, 302)
            self.assertIn(username.encode(), client.get('/').data)
            self.assertEqual(client.get('/logout').status_code, 405)
            self.assertEqual(post('/logout').status_code, 302)
            self.assertEqual(client.get('/').status_code, 302)
        finally:
            with app.app_context():
                with get_db().cursor() as cursor:
                    cursor.execute('DELETE FROM `user` WHERE username=%s', (username,))


if __name__ == '__main__':
    unittest.main()
