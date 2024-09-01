const admin = require('firebase-admin');
const serviceAccount = require('../config/test-bc002-firebase-adminsdk-47w0c-20f1ea4f43.json');

if (!admin.apps.length) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
    });
}

const db = admin.firestore();

module.exports = { db };
