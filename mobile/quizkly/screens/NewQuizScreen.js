import React from 'react';
import { ScrollView, Text, StyleSheet, View, TextInput, Button } from 'react-native';
import { QuestionList } from '../components/Questions/QuestionList';

export default class NewQuizScreen extends React.Component {
  static navigationOptions = {
    title: 'Your quiz',
  };

  constructor(props) {
    super(props);
  }


  componentDidMount() {

  }

  state = {
  }

  // <TextInput
  //   style={styles.docTitle}
  //   placeholder="Enter quiz title here!"
  //   value={this.state.docTitle}
  //   onChangeText{(text) => this.setState({docTitle: text})} />

  render() {
    const params = this.props.navigation.state.params;
    console.log(params);
    return (
      <View style={styles.container}>
        <QuestionList list={params.list}/>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 15,
    backgroundColor: '#eee',
    flexDirection: 'column',
  },
});
