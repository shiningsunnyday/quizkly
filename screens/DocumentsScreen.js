import React from 'react';
import { AsyncStorage, ScrollView, StyleSheet, Button, View } from 'react-native';
import { DocumentList } from '../components/DocumentList';

export default class DocumentsScreen extends React.Component {

  static navigationOptions = {
    title: 'Documents',
  };

  async componentDidMount() {

    let docs = await AsyncStorage.getItem('quizzes') || [];

    console.log("DocumentsScreen mounted, docs is of length", docs.length);

    docs = await JSON.parse(docs);
    this.setState({docs}, function() {
      console.log("Docs got set in componentDidMount")
    });

  }

  state = {
    docs: [],
  }

  navIntoBar = ({item}) => {
    console.log("Got item", item.docTitle, "and about to really nav into it");
    this.props.navigation.navigate('OldQuiz', params = {
      docTitle: item.docTitle,
      text: item.text,
      list: item.list,
    });
  }

  render() {

    console.log("About to render document list");
    console.log("Documents in state is of size", this.state.docs.length);
    return (
      <View style={styles.container}>
        <DocumentList navIntoDoc={this.navIntoBar} docs={this.state.docs}/>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  questionList: {
    backgroundColor: 'red',
    flex: 1,
  },
  container: {
    flex: 1,
    backgroundColor: '#ccc',
  },
});
